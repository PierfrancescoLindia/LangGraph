
# Importiamo TypedDict per definire la struttura dello stato del grafo
from typing import TypedDict

# Importiamo json perché il classificatore LLM restituirà una stringa JSON
# e noi dovremo convertirla in dizionario Python
import json

# Importiamo ChatOllama per usare un modello locale tramite Ollama
from langchain_ollama import ChatOllama

# Importiamo StateGraph per costruire il grafo LangGraph
# START rappresenta il punto iniziale del grafo
# END rappresenta il punto finale del grafo
from langgraph.graph import StateGraph, START, END


# Definiamo la struttura dello stato che passerà da un nodo all'altro
class Stato_Grafo(TypedDict):
    # Richiesta originale scritta dall'utente
    richiesta: str

    # Categoria scelta dal classificatore LLM
    categoria: str

    # Livello di sicurezza della classificazione, da 0 a 1
    confidence: float

    # Breve spiegazione del perché il modello ha scelto quella categoria
    motivazione: str

    # Risposta finale generata dal nodo specializzato
    risposta: str


# Creiamo il modello locale Ollama che useremo nei nodi
llm = ChatOllama(model="llama3.2")


# Nodo classificatore: decide a quale categoria appartiene la richiesta
def nodo_llm_classifier(state: Stato_Grafo) -> Stato_Grafo:
    # Costruiamo il prompt da inviare al modello
    prompt = f"""
    Sei un classificatore di richieste di Data Science.

    Devi classificare la richiesta dell'utente in UNA sola categoria tra:
    - preprocessing
    - modellazione
    - visualizzazione
    - generale

    Devi rispondere SOLO con un JSON valido nel seguente formato:

    {{
      "categoria": "preprocessing",
      "confidence": 0.85,
      "motivazione": "Breve spiegazione della scelta."
    }}

    Regole:
    - categoria deve essere solo una tra: preprocessing, modellazione, visualizzazione, generale.
    - confidence deve essere un numero tra 0 e 1.
    - motivazione deve essere breve.
    - Non aggiungere testo fuori dal JSON.

    Richiesta utente:
    {state["richiesta"]}
    """

    # Inviamo il prompt al modello LLM
    risposta_llm = llm.invoke(prompt)

    # Proviamo a interpretare la risposta del modello come JSON
    try:
        # Convertiamo la stringa JSON prodotta dall'LLM in un dizionario Python
        parsed = json.loads(risposta_llm.content)

        # Cerca parsed["categoria"].
        # Se esiste, usa quel valore.
        # Se non esiste, usa "generale".
        # Poi trasforma tutto in minuscolo.
        categoria = parsed.get("categoria", "generale").lower()

        # Cerca parsed["confidence"].
        # Se esiste, usa quel valore.
        # Se non esiste, usa 0.0.
        # Poi converte il valore in numero decimale.
        confidence = float(parsed.get("confidence", 0.0))

        # Cerca parsed["motivazione"].
        # Se esiste, usa quel valore.
        # Se non esiste, usa una stringa vuota.
        motivazione = parsed.get("motivazione", "")

        # Definiamo le uniche categorie accettate dal nostro grafo
        categorie_valide = [
            "preprocessing",
            "modellazione",
            "visualizzazione",
            "generale",
        ]

        # Controlliamo se la categoria prodotta dal modello è valida
        if categoria not in categorie_valide:
            # Se non è valida, forziamo la categoria a "generale"
            categoria = "generale"

            # Impostiamo confidence a 0 perché non ci fidiamo della classificazione
            confidence = 0.0

            # Salviamo una motivazione tecnica dell'errore
            motivazione = "Categoria non valida prodotta dal modello."

    # Se il modello non restituisce JSON valido, entriamo qui
    except Exception:
        # Usiamo "generale" come categoria di fallback
        categoria = "generale"

        # Confidence a 0 perché il parsing è fallito
        confidence = 0.0

        # Motivazione tecnica dell'errore
        motivazione = "Errore nel parsing del JSON prodotto dal modello."

    # Restituiamo lo stato aggiornato
    return {
        # Manteniamo la richiesta originale dell'utente
        "richiesta": state["richiesta"],

        # Salviamo la categoria scelta dal classificatore
        "categoria": categoria,

        # Salviamo il livello di confidenza
        "confidence": confidence,

        # Salviamo la motivazione della scelta
        "motivazione": motivazione,

        # La risposta finale non è ancora stata generata
        "risposta": state["risposta"],
    }


# Nodo specializzato per richieste di preprocessing
def nodo_preprocessing(state: Stato_Grafo) -> Stato_Grafo:
    # Prompt per rispondere come esperto di preprocessing dati
    prompt = f"""
    Sei un assistente esperto di preprocessing dati.

    Rispondi alla richiesta dell'utente spiegando:
    - il problema
    - le tecniche principali
    - un esempio pratico se utile

    Richiesta:
    {state["richiesta"]}
    """

    # Chiamata al modello LLM
    risposta_llm = llm.invoke(prompt)

    # Restituiamo lo stato aggiornando solo la risposta
    return {
        "richiesta": state["richiesta"],
        "categoria": state["categoria"],
        "confidence": state["confidence"],
        "motivazione": state["motivazione"],
        "risposta": risposta_llm.content,
    }


# Nodo specializzato per richieste di modellazione Machine Learning
def nodo_modellazione(state: Stato_Grafo) -> Stato_Grafo:
    # Prompt per rispondere come esperto di Machine Learning
    prompt = f"""
    Sei un assistente esperto di Machine Learning.

    Rispondi alla richiesta dell'utente spiegando:
    - quale approccio usare
    - quali modelli considerare
    - quali metriche valutare
    - eventuali rischi o attenzioni

    Richiesta:
    {state["richiesta"]}
    """

    # Chiamata al modello LLM
    risposta_llm = llm.invoke(prompt)

    # Restituiamo lo stato aggiornando solo la risposta
    return {
        "richiesta": state["richiesta"],
        "categoria": state["categoria"],
        "confidence": state["confidence"],
        "motivazione": state["motivazione"],
        "risposta": risposta_llm.content,
    }


# Nodo specializzato per richieste di visualizzazione dati
def nodo_visualizzazione(state: Stato_Grafo) -> Stato_Grafo:
    # Prompt per rispondere come esperto di data visualization
    prompt = f"""
    Sei un assistente esperto di data visualization.

    Rispondi alla richiesta dell'utente spiegando:
    - quale grafico usare
    - perché è adatto
    - un esempio pratico se utile

    Richiesta:
    {state["richiesta"]}
    """

    # Chiamata al modello LLM
    risposta_llm = llm.invoke(prompt)

    # Restituiamo lo stato aggiornando solo la risposta
    return {
        "richiesta": state["richiesta"],
        "categoria": state["categoria"],
        "confidence": state["confidence"],
        "motivazione": state["motivazione"],
        "risposta": risposta_llm.content,
    }


# Nodo generale per richieste non specialistiche
def nodo_generale(state: Stato_Grafo) -> Stato_Grafo:
    # Prompt per rispondere come assistente generale di Data Science
    prompt = f"""
    Sei un assistente generale di Data Science.

    Rispondi in modo chiaro, didattico e sintetico.

    Richiesta:
    {state["richiesta"]}
    """

    # Chiamata al modello LLM
    risposta_llm = llm.invoke(prompt)

    # Restituiamo lo stato aggiornando solo la risposta
    return {
        "richiesta": state["richiesta"],
        "categoria": state["categoria"],
        "confidence": state["confidence"],
        "motivazione": state["motivazione"],
        "risposta": risposta_llm.content,
    }


# Nodo usato quando il classificatore non è abbastanza sicuro
def nodo_chiarimento(state: Stato_Grafo) -> Stato_Grafo:
    # Costruiamo una risposta che chiede chiarimento all'utente
    risposta = f"""
    Non sono abbastanza sicuro di quale area della Data Science sia coinvolta.

    Categoria stimata: {state["categoria"]}
    Confidenza: {state["confidence"]}
    Motivazione: {state["motivazione"]}

    Puoi specificare meglio se la tua richiesta riguarda:
    - preprocessing dei dati
    - modellazione Machine Learning
    - visualizzazione dati
    - concetti generali di Data Science
    """

    # Restituiamo lo stato aggiornando la risposta
    return {
        "richiesta": state["richiesta"],
        "categoria": state["categoria"],
        "confidence": state["confidence"],
        "motivazione": state["motivazione"],
        "risposta": risposta,
    }


# Funzione di routing: decide quale nodo eseguire dopo il classificatore
def scegli_percorso(state: Stato_Grafo) -> str:
    # Se la confidence è troppo bassa, mandiamo l'utente al nodo di chiarimento
    if state["confidence"] < 0.65:
        return "chiarimento"

    # Se la categoria è preprocessing, andiamo al nodo preprocessing
    if state["categoria"] == "preprocessing":
        return "preprocessing"

    # Se la categoria è modellazione, andiamo al nodo modellazione
    elif state["categoria"] == "modellazione":
        return "modellazione"

    # Se la categoria è visualizzazione, andiamo al nodo visualizzazione
    elif state["categoria"] == "visualizzazione":
        return "visualizzazione"

    # In tutti gli altri casi, andiamo al nodo generale
    else:
        return "generale"


# Creiamo il costruttore del grafo LangGraph
graph_builder = StateGraph(Stato_Grafo)

# Aggiungiamo il nodo classificatore al grafo
graph_builder.add_node("LLMClassifier", nodo_llm_classifier)

# Aggiungiamo il nodo preprocessing al grafo
graph_builder.add_node("NodoPreprocessing", nodo_preprocessing)

# Aggiungiamo il nodo modellazione al grafo
graph_builder.add_node("NodoModellazione", nodo_modellazione)

# Aggiungiamo il nodo visualizzazione al grafo
graph_builder.add_node("NodoVisualizzazione", nodo_visualizzazione)

# Aggiungiamo il nodo generale al grafo
graph_builder.add_node("NodoGenerale", nodo_generale)

# Aggiungiamo il nodo chiarimento al grafo
graph_builder.add_node("NodoChiarimento", nodo_chiarimento)

# Definiamo che il grafo parte dal nodo LLMClassifier
graph_builder.add_edge(START, "LLMClassifier")

# Definiamo gli edge condizionali dopo il classificatore
graph_builder.add_conditional_edges(
    # Nodo da cui parte la scelta condizionale
    "LLMClassifier",

    # Funzione che decide quale strada prendere
    scegli_percorso,

    # Mappa tra valore restituito da scegli_percorso e nodo successivo
    {
        "preprocessing": "NodoPreprocessing",
        "modellazione": "NodoModellazione",
        "visualizzazione": "NodoVisualizzazione",
        "generale": "NodoGenerale",
        "chiarimento": "NodoChiarimento",
    },
)

# Dopo il nodo preprocessing il grafo termina
graph_builder.add_edge("NodoPreprocessing", END)

# Dopo il nodo modellazione il grafo termina
graph_builder.add_edge("NodoModellazione", END)

# Dopo il nodo visualizzazione il grafo termina
graph_builder.add_edge("NodoVisualizzazione", END)

# Dopo il nodo generale il grafo termina
graph_builder.add_edge("NodoGenerale", END)

# Dopo il nodo chiarimento il grafo termina
graph_builder.add_edge("NodoChiarimento", END)

# Compiliamo il grafo rendendolo eseguibile
grafo = graph_builder.compile()


# Chiediamo all'utente una richiesta di Data Science
richiesta_utente = input("Inserisci una richiesta di Data Science: ")

# Eseguiamo il grafo passando lo stato iniziale
result = grafo.invoke({
    # La richiesta inserita dall'utente
    "richiesta": richiesta_utente,

    # Categoria inizialmente vuota
    "categoria": "",

    # Confidence inizialmente pari a 0
    "confidence": 0.0,

    # Motivazione inizialmente vuota
    "motivazione": "",

    # Risposta inizialmente vuota
    "risposta": "",
})


# Stampiamo la categoria individuata
print("\nCategoria individuata:")
print(result["categoria"])

# Stampiamo il livello di confidenza
print("\nConfidence:")
print(result["confidence"])

# Stampiamo la motivazione della classificazione
print("\nMotivazione:")
print(result["motivazione"])

# Stampiamo la risposta finale
print("\nRisposta:")
print(result["risposta"])
