# RAG from Scratch: Cricket Notes Q&A

A small question-answering system that reads a chapter on cricket and answers questions from it.
The search part is written from scratch in plain Python. An LLM is used only for the last step, to write the final answer.

## What it does

You ask a question. The system finds the 3 most useful pieces of the chapter, then gives them to an LLM, which writes an answer using only those pieces. If the answer is not in the notes, the LLM is told to say it does not know.

This idea is called **RAG** (Retrieval-Augmented Generation):
- **Retrieval:** find the right pieces of text. (Built from scratch here.)
- **Generation:** an LLM writes the answer from those pieces. (Uses the Groq API.)

## How it works

1. **Chunking:** the chapter is cut into pieces of 50 words, and each piece repeats the last 10 words of the one before it, so no sentence is lost at the edge.
2. **TF-IDF search (written by me):** every chunk gets a score for each word. Rare words score high, common words score low. A chunk wins when it has the rare, important words from the question.
3. **Embedding search:** a free model (`all-MiniLM-L6-v2`) turns each chunk and the question into numbers that capture meaning. This finds chunks even when the words are different (for example "stop cheating" and "fraud").
4. **Hybrid search:** both scores are squeezed into the range 0 to 1 and mixed with a weight. Weight 1 means TF-IDF only, weight 0 means embeddings only.
5. **Answer:** the top 3 chunks and the question are put into a prompt, and a Llama model on Groq writes the answer.

## Files

| File | What it is |
|---|---|
| `RAG_Implementation.ipynb` | The notebook with all the code |
| `cricket_chapter.txt` | The cricket chapter used as notes (2,337 words, 59 chunks) |
| `test_set.py` | 50 test questions, each with a phrase that must appear in the correct chunk |

## Test

50 questions in three styles:
- 20 clean questions ("what is a yorker?")
- 15 messy questions with typos or short wording ("wat is a no ball")
- 15 questions that use different words from the notes ("why is bowling harder in the evening")

A search gets a point if the correct chunk is **first** (top-1) or **among the first three** (top-3).

## Results

| Search | Top-1 | Top-3 |
|---|---|---|
| TF-IDF | 66% | 94% |
| Embedding | 68% | 94% |
| Hybrid (weight 0.5) | 76% | 100% |

Trying different hybrid weights (weight is the share given to TF-IDF):

| Weight | Top-1 | Top-3 |
|---|---|---|
| 0 (embeddings only) | 68% | 94% |
| 0.2 | 84% | 98% |
| 0.4 | 80% | 100% |
| 0.6 | 74% | 100% |
| 0.8 | 72% | 98% |
| 1 (TF-IDF only) | 66% | 94% |

What this shows:
- Mixing both searches beat either one alone.
- The answer step reads 3 chunks, so top-3 matters most. Weight 0.4 gives 100% top-3, so the final answer step uses 0.4.
- On this small chapter, embeddings were only slightly better than TF-IDF at first place.

## What I learned

- **Punctuation can break a search.** At first, "what is run out?" with a question mark gave a different result from "what is run out", because the `?` stayed attached to the last word. Cleaning the words with a pattern that keeps only letters and digits fixed it.
- **The model type matters.** One model I tried (a prompt-safety model) only returned a number, not an answer. A small model I tried later gave a wrong answer and ignored the "use only the notes" rule. Choosing a proper chat model is part of the work.
- **A higher score is not a correct answer.** A tiny question can get a high similarity score from the wrong chunk, so the test checks whether the right chunk was found, not how high the score is.

## Limits

- One chapter and 50 questions is a small test. The numbers show a trend, not a final proof.
- The questions were written by one person. Questions from other people would make the results more reliable.
- The LLM answers were checked by reading a few examples, not scored on the full test set.
- TF-IDF still picks the wrong chunk for some questions, for example it ranks a "free hit" chunk above the definition of "run out".


## Tech used

Python, NumPy, `sentence-transformers` (embeddings), Groq API (LLM), Google Colab.
