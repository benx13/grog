from wtpsplit import SaT
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


sat = SaT("sat-3l-sm").half().to('mps')
# optionally run on GPU for better performance
# also supports TPUs via e.g. sat.to("xla:0"), in that case pass `pad_last_batch=True` to sat.split

r = sat.split("This is a test This is another test.")
# returns ["This is a test ", "This is another test."]

print(r)