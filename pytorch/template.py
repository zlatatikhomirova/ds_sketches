"""
Этапы рабочего процесса PyTorch

1. Подготовка данных
2. Построение модели (build or pick a pretrained model)
  2.1. Выбрать loss function и optimizer
  2.2. Построить цикл обучения (training loop)
3. Обучение модели (предсказание можно внести в этот этап,
можно в следующий).
4. Предсказание и оценка модели.
5. Улучшаем модель, корректируя параметры.
6. Сохранение и загрузка модели (save and reload).
"""

# Импорты

import torch
from torch import nn 
from sklearn.model_selection import train_test_split

# Check PyTorch version
print("PyTorch version", torch.__version__)

## 1. Подготовка данных

# data -> torch.Tensor
def prepare_data() -> torch.Tensor:
  ...

# Разбиение данных на train, valid, test
"""
train 
  -	Модель обучается на этих данных	
  - ~60-80%	исходных данных
  - обязательная часть разбиения
valid	
  - Модель настраивает свои гиперпараметры на этих данных.
  - ~10-20%	исходных данных
  - необязательная часть разбиения (но часто используется)
test	
  - Модель оценивается на этих данных, чтобы проверить качество обучения.	
  - ~10-20%	исходных данных
  - обязательная часть разбиения
"""

def split_data(data: torch.Tensor, 
               train_size: float|None=None, 
               valid_split: float|None=None,
               test_size: float|None=None,
               random_state: int=42) -> tuple[torch.Tensor]:
  """
  """
  if train_size == None:
    assert test_size != None, "test and train sizes are not specified."
  else:
    test_size = round(1 - train_size, 4)
  X_train, X_test, y_train, y_test = train_test_split(data,
                                                      test_size=test_size,
                                                      random_state=random_state)
  if valid_split:
    X_valid, X_test, y_valid, y_test = train_test_split(data,
                                                        test_size=valid_split,
                                                        random_state=random_state)
    return X_train, X_vaild, X_test, y_train, y_valid, y_test
  return X_train, X_test, y_train, y_test

## 2. Построение модели

class LinearRegressionModel(nn.Module): # <- практически все в PyTorch это nn.Module (можно представлять это как конструктор)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
                                   requires_grad=True) # <- для возможности вычисления градиентов

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                                requires_grad=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias # <- линейная регрессия (y = m*x + b)

Строительные блоки в PyTorch
В PyTorch есть примерно четыре основных модуля, с помощью которых можно создать практически любую нейронную сеть, какую только можно себе представить.

Это torch.nn, torch.optim, torch.utils.data.Dataset and torch.utils.data.DataLoader. Сейчас мы сосредоточимся на первых двух, а к остальным перейдем позже.

Модуль PyTorch	Что он делает?
torch.nn	Содержит все строительные блоки для вычислительных графов (по сути, серии вычислений, выполняемых определенным образом).
torch.nn.Parameter	Хранит тензоры, которые могут быть использованы с nn.Module. Если requires_grad=True, то градиенты (используемые для обновления параметров модели с помощью градиентного спуска вычисляются автоматически, что часто называют автоматическим дифференцированием.
torch.nn.Module	Базовый класс для всех модулей нейронных сетей, все строительные блоки для нейронных сетей являются подклассами. Если вы строите нейронную сеть в PyTorch, ваши модели должны быть подклассами nn.Module. Требует реализации метода forward().
torch.optim	Содержит различные алгоритмы оптимизации (они указывают параметрам модели, хранящимся в nn.Parameter, как лучше сдвинуться чтобы улучшить качество предсказаний модели).
def forward()	Все подклассы nn.Module требуют метод forward(), который определяет вычисления, которые будут происходить на данных, переданных конкретному nn.Module (например, как в классе линейной регрессии выше).
