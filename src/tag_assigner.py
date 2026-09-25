import re # Per l'analisi lessicale dell'Overlap Coefficient
import sys
from pathlib import Path

# Setup root progetto
project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langid.langid import LanguageIdentifier, model as LANGID_MODEL

from src.config import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_THREAD,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_URL,
    TAG_ASSIGN_THRESHOLD,
    TAG_WEIGHT_COSINE,
    TAG_WEIGHT_OVERLAP,
)

import numpy as np
from typing import Dict, List, Optional, Union, Set, Tuple

from tqdm import tqdm

# Stopword da escludere nel calcolo dell'Overlap lessicale
#  Serve a impedire che semplici connettivi o articoli gonfino il punteggio
#  di similarità. Copre solo ITA e INGLESE
STOPWORDS: Set[str] = {
    # Italiano
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
    "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "che", "chi",
    "cui", "non", "più", "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle", "dal", "dallo", "dalla",
    "dai", "dagli", "dalle", "nel", "nello", "nella", "nei", "negli",
    "nelle", "sul", "sullo", "sulla", "sui", "sugli", "sulle", "questo",
    "questa", "questi", "queste", "quello", "quella", "sono", "sia",
    "stato", "è", "ed", "ad",
    # Inglese
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "this", "that",
}

# Del fallback di expand_tag(): sono vocaboli di ripiego quando l'espanisone fallisce, vengono quindi eliminati
FALLBACK_TEMPLATE_TOKENS: Set[str] = {"argomenti", "sinonimi", "concetti", "correlati"}

# Per il matching lessicale è necessario che l'espansione dei tag venga effettuata nella lingua del testo fornito
_LANGUAGE_IDENTIFIER = LanguageIdentifier.from_modelstring(LANGID_MODEL, norm_probs=True)
# Soglia sotto cui l'identificazione è inaffidabile. Sotto di essa si ricade sul default value
MIN_CHARS_FOR_LANG_DETECTION = 20
SUPPORTED_LANGUAGES: Set[str] = {"it", "en"} # Le uniche lingue possibilmente riconoscibili

def detect_language(text: str, default: str = "it") -> str:
    """
    Rileva la lingua del testo (codice ISO 639-1) tramite langid. Per testi troppo corti
    la rilevazione non è affidabile o se la lingua non rientra tra quelle supportate, si forza il fallback
    sulla lingua di default.
    """
    if not text or len(text.strip()) < MIN_CHARS_FOR_LANG_DETECTION:
        return default

    lang, _confidence = _LANGUAGE_IDENTIFIER.classify(text)

    if lang not in SUPPORTED_LANGUAGES:
        return default

    return lang

class TagAssigner:
    """
    Assegnatore ibrido deterministico-llm per l'assegnazione dei tag forniti
    inerenti al testo fornito.
    Usato sia nel TagReranker che per il Pre-Tagging, è fatto in modo da poter gestire entrambe le situazioni

    Supporta testi in lingue diverse, assegnando correttamente il tag indipendentemente dalle due
    """

    def __init__(
        self,
        threshold: float = TAG_ASSIGN_THRESHOLD,
        weight_cosine: float = TAG_WEIGHT_COSINE,
        weight_overlap: float = TAG_WEIGHT_OVERLAP,
        embedding_model_name: str = EMBEDDING_MODEL,
        llm_model_name: str = LLM_MODEL,
        ollama_url: str = OLLAMA_URL,
        default_language: str = "it",
    ):
        self.threshold = threshold
        self.weight_cosine = weight_cosine
        self.weight_overlap = weight_overlap
        self.default_language = default_language

        self.embedding_model = FastEmbedEmbeddings(model_name=embedding_model_name)
        self.llm = ChatOllama(
            model=llm_model_name,
            base_url=ollama_url,
            keep_alive=DEFAULT_KEEP_ALIVE,
            num_thread=DEFAULT_NUM_THREAD,
            temperature=0,
        )

        # Dizionari per memorizzare l'espansione dei tag e i relativi
        #  vettori, risparmiando il ricalcolo ogni volta e permettendo
        #  il riutilizzo degli stessi tag_espansi per tutta l'esecuzione
        #  dipende ovviamente dalla singola istanza della classe, quindi
        #  diviene fondamentale non reistanziarla fintanto che si vuole costanza nelle frasi espansione.
        #  Hanno come chiavi (tag, language): espansione ed embedding cambiano a seconda della lingua del testo analizzato,
        #  ai fini del riconoscimento lessicale (overlap). Quello semantico, basato su cos_sim, si affida alla gestione multilingua dell'embedding model
        self.expanded_tags_cache: Dict[Tuple[str, str], str] = {}
        self.tag_embeddings_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._fallback_tags: Set[Tuple[str, str]] = set()

    def get_embedding(self, text: str) -> np.ndarray:
        """Restituisce il vettore di embedding"""
        return np.array(self.embedding_model.embed_query(text), dtype=float)

    def expand_tag(self, tag: str, lang: str) -> str:
        """
        Genera una frase descrittiva non ambigua estesa per il tag tramite ChatOllama e salva in cache per riusi futuri.
        Generazione e salvataggio in cache di un tag sono differenziati per lingua del testo analizzato
        """
        cache_key = (tag, lang)
        if cache_key in self.expanded_tags_cache:
            return self.expanded_tags_cache[cache_key]

        prompt = (
            f"Descrivi il tag '{tag}' in lingua '{lang}' in una singola frase densa di informazioni.\n"
            f"Includi obbligatoriamente: sinonimi diretti, sotto-categorie principali, strumenti o componenti chiave, ed entità o termini tecnici correlati.\n"
            f"Evita preamboli da dizionario (es. 'È un'attività che...') e concentrati sull'inserire il maggior numero di sostantivi specifici del settore.\n"
            f"Formato: '{tag}: <frase>'. Senza virgolette."
            """
            f"Fornisci un'espansione descrittiva per il tag '{tag}' in lingua '{lang}'.\n"
            f"Includi concetti generali ampiamente noti, sinonimi e plurali (anche in inglese se diffusi).\n"
            f"Concentrati esclusivamente su sostantivi e concetti chiave, evitando verbi inutili e termini gergali inventati.\n"
            f"Formato tassativo: '{tag}: <testo_espanso>'.\n"
            f"Non usare virgolette e rispondi solo con la riga richiesta."
            """
            """
            f"Genera una singola e breve frase descrittiva per il tag '{tag}', "
            f"scritta ESCLUSIVAMENTE nella lingua con codice ISO 639-1 '{lang}'. "
            f"La frase deve ripetere il tag e dopo : contenere il concetto esteso, sinonimi e termini chiave correlati in quella lingua,"
            f"evitando termini ambigui e inserendo anche i plurali dei termini più significativi e eventuali termini tecnici in inglese e nella lingua del codice ISO. "
            f"Rispondi ESCLUSIVAMENTE con il tag ripetuto seguito dalla frase descrittiva, senza alcun testo aggiuntivo."
            """
        )

        try:
            raw_response = self.llm.invoke(prompt).content.strip()
            if raw_response.startswith(f"{tag}:"): # Per garantire che inizi ripetendo il tag all'inizio
                expanded_text = raw_response
            else:
                expanded_text = f"{tag}: {raw_response}"
        except Exception as e: #frase di fallback
            expanded_text = f"{tag}: argomenti, sinonimi e concetti correlati a {tag}"
            self._fallback_tags.add(cache_key) # Esclusi anche i vocaboli del fallback
            print(f"!!!>>> Errore nell'espansione del tag '{tag}': {e}")

        self.expanded_tags_cache[cache_key] = expanded_text

        ### DEBUG
        print(f"pDB>> tag: {tag}")
        print(f"pDB>> expanded_tag:\n  >>>{expanded_text}")
        ######

        return expanded_text

    def cosine_sim(self, v1, v2):
        """Calcola la cosine_similarity tra due vettori"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1,v2) / (norm1*norm2))

    def compute_overlap(
        self,
        text: str,
        expanded_tag: str,
        tag: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> float:
        """
        Calcola quanta parte del vocabolario del tag espanso è presente nel testo.
        Si divide intenzionalmente per la sola
        dimensione del tag (differente da "similarità di Jaccard"), per evitare che un testo molto più lungo
        diluisca artificialmente il punteggio. Le stopword vengono escluse da entrambi gli insiemi prima del confronto.
        """
        tokens_text = set(re.findall(r"\b\w+\b", text.lower())) - STOPWORDS
        tokens_tag = set(re.findall(r"\b\w+\b", expanded_tag.lower())) - STOPWORDS

        # Se si è ricaduti nel fallback, dal tag esteso si tolgono i vocaboli relativi
        if tag is not None and lang is not None and (tag, lang) in self._fallback_tags:
            tokens_tag -= FALLBACK_TEMPLATE_TOKENS

        if not tokens_tag:
            return 0.0

        return len(tokens_text.intersection(tokens_tag)) / len(tokens_tag)


    def assign_tags(
        self,
        text: str,
        candidate_tags: List[str],
        vector: Optional[Union[List[float], np.ndarray]] = None, # rende opzionale, inoltre permette sia ndarray che lista di float
    ) -> List[str]:
        """
        Analizza il testo e restituisce la lista dei soli tag che superano una certa soglia di similarità.
        Lo score comparato con la soglia è calcolato con cos_sim e overlap lessicale pesate tra loro su un tag espanso tramite llm
        Viene individuata la lingua del test per sapere come espandere il tag per la somiglianza lessicale
        """
        lang = detect_language(text, default=self.default_language)

        # se passato il vettore usa quello, altrimenti lo calcola
        text_vector = self.get_embedding(text) if vector is None else vector

        assigned_tags = []

        for tag in candidate_tags:            
            expanded_tag_str = self.expand_tag(tag, lang)
            cache_key = (tag, lang)

            if cache_key not in self.tag_embeddings_cache:
                self.tag_embeddings_cache[cache_key] = self.get_embedding(expanded_tag_str)
            tag_vector = self.tag_embeddings_cache[cache_key]

            score_cs = self.cosine_sim(text_vector, tag_vector)

            score_overlap = self.compute_overlap(text, expanded_tag_str, tag=tag, lang=lang)

            final_score = (self.weight_cosine * score_cs) + (self.weight_overlap * score_overlap)

            if final_score >= self.threshold:
                assigned_tags.append(tag)

        return assigned_tags
