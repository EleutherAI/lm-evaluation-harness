
def doc_to_text(x):

    verbalized_rebus = x["verbalized_rebus"]
    solution_key = x["solution_key"]

    PROMPT_TEMPLATE = """Sei un'esperto risolutore di giochi enigmistici. Il seguente gioco contiene una frase (Rebus) nella quale alcune parole sono state sostituite da indizi tra parentesi quadre. Il tuo compito è quello di identificare le parole nascoste e sostituirle agli indizi nel Rebus, producendo una prima lettura dalla quale poi si deriverà una frase risolutiva. La chiave di lettura è una sequenza di numeri che rappresentano la rispettive lunghezze delle parole che compongono la frase risolutiva. La tua risposta deve essere una frase risolutiva sensata e che rispetti le lunghezze definite nella chiave di lettura.
                    # Esempio 1:
                    Rebus: AC [Un mollusco nell'insalata di mare] GLI [Lo è l'operaio che lavora in cantiere] S TO [Soldati da trincea]
                    Chiave di lettura: 11 2 10
                    Procediamo alla risoluzione del rebus passo per passo:
                    - A C = A C
                    - [Un mollusco nell'insalata di mare] = cozza
                    - G L I = G L I
                    - [Lo è l'operaio che lavora in cantiere] = edile
                    - S T O = S T O
                    - [Soldati da trincea] = fanti
                    Prima lettura: AC cozza GLI edile S TO fanti
                    Ora componiamo la soluzione seguendo la chiave risolutiva:
                    11 = Accozzaglie
                    2 = di
                    10 = lestofanti
                    Soluzione: Accozzaglie di lestofanti
                    # Esempio 2:
                    Rebus: [Edificio religioso] G [Lo fa doppio l'opportunista] NP [Poco cortese, severo] NZ [Parente... molto lontana]
                    Chiave di lettura: 3 1 6 3 8 2
                    Procediamo alla risoluzione del rebus passo per passo:
                    - [Edificio religioso] = chiesa
                    - G = G
                    - [Lo fa doppio l'opportunista] = gioco
                    - N P = N P
                    - [Poco cortese, severo] = rude
                    - N Z = N Z
                    - [Parente... molto lontana] = ava
                    Prima lettura: chiesa G gioco NP rude NZ ava
                    Ora componiamo la soluzione seguendo la chiave risolutiva:
                    3 = Chi
                    1 = è
                    6 = saggio
                    3 = con
                    8 = prudenza
                    2 = va
                    Soluzione: Chi è saggio con prudenza va
                    # Ora tocca a te!
                    Completa il rebus seguendo il procedimento descritto, rispondendo esattamente nello stesso formato utilizzato dagli esempi precedenti.
                    Rebus: {{verbalized_rebus}}
                    Chiave di lettura: {{solution_key}}
                    """

    instance = PROMPT_TEMPLATE.format(verbalized_rebus=x["verbalized_rebus"], solution_key=x["solution_key"])

    return instance

