import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

x = torch.linspace(-5, 5, 1000)
gelus = nn.GELU()
y = gelus(x)

plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y.numpy(), label='GELU')
plt.title('GELU 활성화 함수')
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.grid(True)
plt.legend()
plt.show() 