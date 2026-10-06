"""
Long-term memory: information and experiences kept OUTSIDE the neural network.

    BRAIN   = learned general language ability (brain/): its weights never change here
    MEMORY  = what you told it, stored on this Mac (memory/, state/memory.db)
    CHAT    = the interaction between the two (chat/)

store.py      the SQLite database: memories, conversation history, an audit log
writer.py     decides what is worth remembering (only YOUR messages; never the model's)
retrieval.py  scores memories for relevance to a new message
system.py     connects memory to the chat: forget -> retrieve -> (brain) -> form -> store
evaluate.py   the memory benchmark (retrieval and generation measured separately)

Nothing here writes a reply, and nothing here tells the system what it is.
"""
