import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')


def gpt3(prompt, engine='davinci', response_length=64,
         temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0,
         start_text='', restart_text='', stop_seq=[]):
    response = openai.Completion.create(
        prompt=prompt + start_text,
        engine=engine,
        max_tokens=response_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop_seq,
    )
    answer = response.choices[0]['text']
    new_prompt = prompt + start_text + answer + restart_text
    return answer, new_prompt


def chat():
    while True:
        prompt = input(' ')
        answer, prompt = gpt3(
            prompt,
            engine='davinci:ft-personal:org-ai-2022-12-19-02-09-27',
            response_length=250,
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
            start_text=f"{prompt}\n\n###\n\n",
            restart_text='\n',
            stop_seq=[' END']
        )
        print(answer)


if __name__ == '__main__':
    chat()
