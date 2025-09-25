from openai import OpenAI

# API anahtarını buraya gir
client = OpenAI(api_key="kendi-secret-key'im")

def chat_with_gpt(prompt, history):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=history + [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    history = []

    while True:
        user_input = input("Mesajınız nedir? ")

        if user_input.lower() in ["exit", ""]:
            print("Görüşme tamamlandı")
            break

        # Kullanıcı mesajını geçmişe ekle
        history.append({"role": "user", "content": user_input})

        # ChatGPT cevabını al
        response = chat_with_gpt(user_input, history)

        # Sohbet geçmişine cevabı ekle
        history.append({"role": "assistant", "content": response})

        print("Chatbot:", response)
