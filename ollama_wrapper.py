from langchain_ollama import OllamaLLM

def get_local_llm():
    llm = OllamaLLM(
        model="qwen2.5",
        base_url="http://localhost:11434"   # Windows Ollama server
    )
    return llm

if __name__ == "__main__":
    llm = get_local_llm()
    print(llm.invoke("What model are you? And which version?"))
