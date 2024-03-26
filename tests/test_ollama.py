from foundationevals.chatbots.ollama import Ollama


def test_can_chat_to_ollama():
    chatbot = Ollama(model="llama2")
    response = chatbot.chat("Hello")
    assert isinstance(response, str)
    assert response
    assert "hi" in response.lower() or "hello" in response.lower()
