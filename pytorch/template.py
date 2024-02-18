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

"""
Строительные блоки в PyTorch
В PyTorch есть примерно четыре основных модуля, с помощью которых
можно создать практически любую нейронную сеть, какую только можно себе представить.

torch.nn	
  - Содержит все строительные блоки для вычислительных графов 
    (по сути, серии вычислений, выполняемых определенным образом).
    * def forward()	
      - Все подклассы nn.Module требуют метод forward(),
        который определяет вычисления, которые будут происходить на данных,
        переданных конкретному nn.Module (например, как в классе линейной регрессии ниже).
torch.nn.Parameter	
  - Хранит тензоры, которые могут быть использованы с nn.Module. Если requires_grad=True,
    то градиенты (используемые для обновления параметров модели с помощью градиентного 
    спуска вычисляются автоматически, что часто называют автоматическим дифференцированием.
torch.nn.Module	
  - Базовый класс для всех модулей нейронных сетей, все строительные блоки для нейронных сетей
    являются подклассами. Если вы строите нейронную сеть в PyTorch, ваши модели должны быть 
    подклассами nn.Module. Требует реализации метода forward().
torch.optim	
  - Содержит различные алгоритмы оптимизации (они указывают параметрам модели, хранящимся
    в nn.Parameter, как лучше сдвинуться чтобы улучшить качество предсказаний модели).

"""

class LinearRegressionModel(nn.Module): # <- практически все в PyTorch это nn.Module (можно представлять это как конструктор)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
                                   requires_grad=True) # <- для возможности вычисления градиентов

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float),
                                requires_grad=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias # <- линейная регрессия (y = m*x + b)

# Проверка содержимого модели
# выставляем manual так как nn.Parameter инициализируются случайно
torch.manual_seed(42)

# создадим экземпляр модели 
model_0 = LinearRegressionModel()

# Проверка содержимого
list(model_0.parameters())

# List named parameters 
model_0.state_dict()

# делаем предсказания
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# так же можно встретить torch.no_grad()
# это старый вариант
# with torch.no_grad():
#   y_preds = model_0(X_test)

## 3. Обучение модели
"""
Чтобы наша модель могла самостоятельно обновлять свои параметры, нам нужно добавить в наш пайплайн еще несколько вещей.

Это функция потерь, а также оптимизатор.

Модуль	Что она делает?	Где она находится в PyTorch?	Общие значения
Функция потерь	Измеряет, насколько ошибочны предсказания вашей модели (например, y_preds) по сравнению с истинными метками (например, y_test). Чем меньше, тем лучше.	В PyTorch есть множество встроенных функций потерь в torch.nn.	Средняя абсолютная ошибка (MAE) для задач регрессии (torch.nn.L1Loss()). Двоичная кросс-энтропия для задач двоичной классификации (torch.nn.BCELoss()).
Оптимизатор	Подсказывает вашей модели, как обновить ее внутренние параметры, чтобы наилучшим образом снизить потери.	Различные реализации функций оптимизации можно найти в torch.optim.	Стохастический градиентный спуск (torch.optim.SGD()). Оптимизатор Adam (torch.optim.Adam()).
Давайте создадим функцию потерь и оптимизатор, которые помогут улучшить нашу модель.

В зависимости от того, над какой проблемой вы работаете, будет зависеть, какую функцию потерь и какой оптимизатор вы используете.

Однако есть несколько общих значений, которые, как известно, хорошо работают, например SGD (стохастический градиентный спуск) или Adam. А также функция потерь MAE (средняя абсолютная ошибка) для задач регрессии (предсказание числа) или функция потерь бинарной кросс-энтропии для задач классификации.
"""

"""
Также, мы будем использовать SGD, torch.optim.SGD(params, lr), где:

params - параметры целевой модели, которые вы хотите оптимизировать (например, значения weights и bias, которые мы произвольно задали ранее).
lr - скорость обучения, с которой оптимизатор должен обновлять параметры. Большее значение означает, что оптимизатор будет пробовать большие обновления
(иногда они могут быть слишком большими и оптимизатор не сработает), меньшее - меньшие обновления (иногда они могут быть слишком маленькими и оптимизатор 
будет слишком долго искать идеальные значения). Скорость обучения считается гиперпараметром (потому что она задается перед обучением модели). 
Обычные начальные значения lr - 0.01, 0.001, 0.0001, однако их также можно регулировать со временем (это называется планирование скорости обучения или шедулинг).
Для нашей задачи, поскольку мы предсказываем число, давайте используем MAE (находится в разделе torch.nn.L1Loss()) в PyTorch в качестве функции потерь.
"""
# Создадим loss функцию
loss_fn = nn.L1Loss() # MAE loss 

# Создадим оптимизатор
optimizer = torch.optim.SGD(params=model_0.parameters(), # передаем параметры модели, которые мы хотим изменять
                            lr=0.01) 
next(model_0.parameters())

## Training loop
"""
Для цикла обучения мы построим следующие шаги:

Номер	Название шага	Что он делает?	Пример кода
1	Forward pass	Модель проходит через все обучающие данные один раз, выполняя вычисления функции forward().	model(x_train)
2	Вычисление потерь	Выходы (предсказания) модели сравниваются с истиными метками и оцениваются на предмет того, насколько они ошибочны.	loss = loss_fn(y_pred, y_train)
3	Нулевые градиенты	Градиенты, хранящиеся в оптимизаторах обнуляются (они накапливаются по умолчанию), чтобы их можно было пересчитать для конкретного шага обучения.	optimizer.zero_grad()
4	Обратное распространение (backward pass)	Вычисляет градиент лосса относительно каждого обновляемого параметра модели (каждый параметр с requires_grad=True). Это известно как обратное распространение, отсюда и "backward".	loss.backward()
5	Обновление оптимизатора (градиентный спуск)	Обновляем параметры с requires_grad=True относительно градиентов лосса, чтобы улучшить их.
"""

"""
PyTorch testing loop
Что касается цикла тестирования (оценки нашей модели), то типичные шаги включают:

Номер	Название шага	Что он делает?	Пример кода
1	Forward pass	Модель проходит через все обучающие данные один раз, выполняя вычисления функции forward().	model(x_test)
2	Вычисление потерь	Выходы (предсказания) модели сравниваются с истиной и оцениваются на предмет того, насколько они ошибочны.	loss = loss_fn(y_pred, y_test)
3	Расчет метрик оценки (необязательно)	Наряду со значением потерь вы можете рассчитать другие метрики оценки, например accuracy на тестовом наборе.	Пользовательские функции
Обратите внимание, что цикл тестирования не содержит выполнения обратного распространения (loss.backward()) или шага оптимизатора (optimizer.step()),
это потому, что никакие параметры в модели не изменяются во время тестирования, они уже вычислены. Для тестирования нас интересует только результат прямого прохода по модели.
"""

"""
Давайте соберем все вышесказанное вместе и обучим нашу модель в течение 100 эпох (прямых проходов по данным) и будем оценивать ее каждые 10 эпох.
"""

torch.manual_seed(42)

epochs = 400

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training
    X_train = X_train.to(device)
    # переводим модель в режим тренировки (он выставляется по умолчанию)
    model_0.train()

    # 1. Прямой проход по данным неявно используя метод forward() 
    y_pred = model_0(X_train)

    # 2. Вычисление лосса 
    loss = loss_fn(y_pred, y_train)

    # 3. Занулим градиенты
    optimizer.zero_grad()

    # 4. Обратный проход
    loss.backward()

    # 5. Сдвигаем параметры модели
    optimizer.step()

    ### Тестирование

    # Переводим модель в режим инференса
    model_0.eval()

    with torch.inference_mode():
      # 1. Прямой проход по тестовым данным
      test_pred = model_0(X_test)

      # 2. Вычисление лосса
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

## визуализация процесса обучения
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

# параметры после обучения
print("Параметры обученной модели:")
print(model_0.state_dict())
print("\nИзначальные параметры:")
print(f"weights: {weight}, bias: {bias}")

"""
Делаем предсказания обученной моделью (инференс)
При предсказании с помощью модели PyTorch необходимо помнить три вещи:

1. Переведите модель в режим оценки (model.eval()).
2. Делайте предсказания, используя контекстный менеджер (with torch.inference_mode(): ...).
3. Все предсказания должны быть сделаны на одном устройстве (например, данные и модель только на GPU или данные и модель только на CPU).
"""

model_0.eval()

with torch.inference_mode():
  # 3. Убедитесь, что вычисления выполняются с моделью и данными на одном и том же устройстве
  # в нашем случае мы еще не настроили код отвечающий за это, поэтому наши данные и модель находятся
  # на процессоре по умолчанию.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds

"""
5. Сохранение и загрузка модели PyTorch
Если вы обучили модель в PyTorch, скорее всего, вы захотите сохранить ее и экспортировать куда-нибудь.

Например, вы можете обучать ее в Google Colab или на локальной машине с GPU, но теперь хотите экспортировать ее в какое-нибудь приложение, где ее смогут использовать другие.

Или, может быть, вы хотите сохранить свой прогресс в работе над моделью и вернуться и загрузить ее позже.

Для сохранения и загрузки моделей в PyTorch существует три основных метода, о которых вы должны знать (можно посмотреть документацию PyTorch save and loading models guide):

Метод PyTorch	Что он делает?
torch.save	Сохраняет сериализованный объект на диск с помощью утилиты Python pickle. Модели, тензоры и различные другие объекты Python, например словари, можно сохранять с помощью torch.save.
torch.load	Использует функцию pickle для десериализации и загрузки в память объектов Python (например, моделей, тензоров или словарей). Вы также можете указать, на какое устройство загружать объект (CPU, GPU и т.д.).
torch.nn.Module.load_state_dict	Загружает словарь параметров модели (model.state_dict()), используя сохраненный объект state_dict().
Сохранение - state_dict()
1. Создадим каталог для сохранения моделей под названием models, используя модуль pathlib.
2. Создадим путь к файлу для сохранения модели.
3. Вызовем команду torch.save(obj, f), где obj - это state_dict() целевой модели, а f - это имя файла, в который нужно сохранить модель.
"""
model_0.state_dict()

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # сохраняем параметры модели
           f=MODEL_SAVE_PATH) 
"""
Загрузка - state_dict()
Поскольку у нас есть сохраненная модель state_dict(), мы можем загрузить ее с помощью torch.nn.Module.load_state_dict(torch.load(f)), где f - путь к файлу нашей сохраненной модели state_dict().

Зачем вызывать torch.load() внутри torch.nn.Module.load_state_dict()?

Поскольку мы сохранили только state_dict(), который является словарем выученных параметров, а не целую модель, мы должны сначала загрузить state_dict() с помощью torch.load(), а затем передать этот state_dict() новому экземпляру нашей модели (который является подклассом nn.Module).
"""
# Создадим новый экземпляр нашей модели
loaded_model_0 = LinearRegressionModel()

# Загрузим state_dict 
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model

# Сравним предыдущие предсказания модели с новой
y_preds == loaded_model_preds

## Расположение данных и моделей
import torch
from torch import nn 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

device

next(loaded_model_0.parameters()).device

X_train.to(device)
X_train.device

## визуализации
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
