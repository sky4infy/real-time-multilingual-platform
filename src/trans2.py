# from src.translate import Translator
from translate import Translator
import time
t = Translator()
# preload first
t.translate('hello', 'en', 'hi')
# now measure pure inference
start = time.time()
for i in range(3):
    r = t.translate('Hello how are you, my name is Akash Kumar', 'en', 'hi')
    print(f"Run {i+1}: {r['latency']}s — {r['translated_text']}")
print(f'Average: {round((time.time()-start)/3, 2)}s')