"""
Learning from experience: what the system GAINED from what happened (memory/ keeps what happened).

    MEMORY    "What happened?"      "On 30 Sep you said: 'I have a dog named Max.'"
    LEARNING  "What is known now?"  your dog --name--> Max;  your dog --is a--> golden retriever

knowledge.py  the knowledge store (SQLite): entities, facts with their history, the learning-experience log
extract.py    reads one of YOUR sentences and finds (subject, relation, value) in it, with readable rules
learner.py    the pipeline: analyze -> novelty -> importance -> store -> connect, and recall for a question
system.py     connects learning to the chat (the same way memory/ does)
evaluate.py   the learning benchmark: the knowledge system and the brain measured separately
import_knowledge.py  general knowledge from files YOU approve (data/knowledge/); nothing is preloaded

The neural network's weights never change here. Nothing here writes a reply, and nothing here tells
the system what it is or what to believe: it records what YOU said, where it came from, how sure to be.
"""
