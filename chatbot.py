from rag_pipeline import answer_query

print("\n=== AI Course Tutor ===\n")
print("Ask me anything about the course. Type 'exit' to quit.\n")

while True:
    q = input("You: ")
    if q.lower() in ("exit", "quit"):
        break

    ans = answer_query(q)
    print("\nTutor:", ans, "\n")
