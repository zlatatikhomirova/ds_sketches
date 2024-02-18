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
