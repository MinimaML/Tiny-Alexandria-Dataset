
# ROLE
You are a Memory Specialist designed to test Long-Context Recall.

# INSTRUCTION
You will be given a text (SOURCE).
Your task is to generate a **Long-Context Retrieval Task**.
1.  Identify a specific, small detail in the text (the "Needle").
2.  Create a Question that requires finding that detail.
3.  Rewrite the text or pad it to ensure the detail is buried in the middle.
4.  The Goal is to force the model to look back and retrieve the exact information.

# FORMAT
<|begin_of_thought|>
(Identify the needle. Plan the distraction/padding text. Formulate the question.)
<|end_of_thought|>

Context:
(The long text with the buried needle)

Question: (The question asking for the needle)
Answer: (The exact detail)

# SOURCE
{text}
