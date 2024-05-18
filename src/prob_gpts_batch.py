import time
import openai
import tiktoken
from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


class Chatbot_gpt3 :
    def __init__(
        self,
        engine: str,
    ) -> None:
        self.engine = engine
        self.last_request_time = 0
        self.request_interval = 1
        self.max_backoff_time = 60

    def calculate_probs(self, prompt):
        prompt = prompt.replace('\x00','')
        is_retry = True

        while is_retry:
            elapsed_time = time.monotonic() - self.last_request_time
            if elapsed_time < self.request_interval:
                time.sleep(self.request_interval - elapsed_time)
            self.last_request_time = time.monotonic()
            try:
                responses = openai.Completion.create(
                            engine=self.engine,
                            prompt=prompt,
                            max_tokens=0,
                            temperature=1.0,
                            logprobs=5,
                            echo=True)
                is_retry = False
            except:
                print("Exceed max tries.")
            
            if is_retry == True:
                self.request_interval *= 2
                if self.request_interval > self.max_backoff_time:
                    self.request_interval = self.max_backoff_time
                print(f"Rate limit hit. Sleeping for {self.request_interval} seconds.")
                time.sleep(self.request_interval)

        data = responses["choices"][0]["logprobs"]
        log_probs = [d for d in data["token_logprobs"] if d is not None]
    
        return log_probs
    
    def reset(self) :
        self.last_request_time = 0
        self.request_interval = 1  # seconds
        self.max_backoff_time = 60  # seconds

class Chatbot_gpt4(Chatbot_gpt3) :
    def get_prob(self, prompt):
        completion = openai.ChatCompletion.create(
            model=self.engine,
            logprobs=True,
            top_logprobs=1,
            max_tokens=1,
            logit_bias=None,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]

    def find_logprob(self, prefix, token_id):
        word = self.encoder.decode([token_id])
        word = word.replace(' ','')
        if word != '':
            token_id = self.encoder.encode(word)[0]
        
        while True :
            try :
                res = self.get_prob(prefix)
                top_logprob = res['logprob']
                if token_id == self.encoder.encode(res['token'])[0]:
                    return res['logprob']
                break
            except : pass

        l, r, eps = -100, 100, 1e-2
        while r-l > eps :
            try :
                mid = (l+r)/2
                res = self.get_prob(prefix)
                if token_id == self.encoder.encode(res['token'])[0] : r = mid
                else : l = mid
            except : pass
                    
        return top_logprob-l
        
    def calculate_probs(self, prompt):
        prompt = prompt.replace('\x00','')
        self.encoder = tiktoken.encoding_for_model(model_name=self.engine)
        encode_prompt = self.encoder.encode(prompt)
        log_probs = []
        for i in range(len(encode_prompt)):
            prefix=self.encoder.decode(encode_prompt[:i])
            log_prob = self.find_logprob(prefix, encode_prompt[i])
            log_probs.append(log_prob)
        return log_probs
    
class ChatbotWrapper :
    def __init__(self, api_key, engine) -> None:
        openai.api_key = api_key
        self.engine = engine
    
    def ask_batch(self, batch_data: list[str], thread_num=1) -> list:
        executor = ThreadPoolExecutor(max_workers=thread_num)
        chatbot_q = Queue(maxsize=thread_num)
        for j in range(thread_num):
            chatbot_q.put(Chatbot_gpt4(engine=self.engine) 
                          if 'gpt-3.5' in self.engine or 'gpt-4' in self.engine 
                          else Chatbot_gpt3(engine=self.engine))
        results = list(tqdm(executor.map(ChatbotWrapper.ask, [chatbot_q for _ in range(len(batch_data))], batch_data), 
                       total=len(batch_data)))
        batch_reponses = []
        for _, res in enumerate(results):
            batch_reponses.append(res)
        return batch_reponses

    @staticmethod
    def ask(chatbot_q: Queue, question:str) -> list:
        if chatbot_q.empty():
            raise Exception("no available chatbot")
        chatbot = chatbot_q.get()
        response = chatbot.calculate_probs(question)
        chatbot_q.put(chatbot)
        return response