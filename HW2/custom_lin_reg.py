import numpy as np
import torch as T


class CustomLogisticRegression:

    def __init__(self, device, num_of_features):
        self.device = device
        self.w: T.tensor = T.zeros((num_of_features), dtype=T.float32, requires_grad=True).to(device)
        self.b: T.tensor = T.zeros((1), dtype=T.float32, requires_grad=True).to(device)
        self.num_of_features = num_of_features
            
    @staticmethod
    def forward(
            x: T.tensor,
            w: T.tensor,
            b: T.tensor):
        """
        Функция для нахождения сигмоиды 
        Args:
        x - вектор признаков
        w - вектор весов
        b - свободный коэффициент
        """

        z = T.dot(x, w).reshape(1)
        z += b
        p = 1 / (1 + T.exp(-z))
        return p

    def random_w_and_b(self):
        """
        Метод для рандомизации (начального задания) коэффициентов w и b
        """

        # границы для случайной величины
        lo = -0.01
        hi = 0.01

        # Начинаем со случайных величин
        # 2 - 2 признака, low, high - границы сверху и снизу
        w = T.rand((self.num_of_features), dtype=T.float32, requires_grad=True).to(self.device)

        # Нормируем на low-high
        w = (hi - lo) * w + lo

        # Зададим градиент
        w.grad = T.zeros(self.num_of_features)
        w.retain_grad()

        # Зададим свобоный член
        b = T.zeros((1), dtype=T.float32, requires_grad=True).to(self.device)
        b.grad = T.zeros(1)
        b.retain_grad()
        return w, b

    def fit(
            self,
            train_x: T.Tensor,
            train_y: T.Tensor,
            lrn_rate: np.double,
            indices: np.array,
            num_of_epoch_iterations: int,
            regular: int = 0
    ) -> T.Tensor:

        """Метод для обучения выборки и нахождения коэффициентов w. Оптимизирующий метод - градиентный спуск
        Args:
        """

        w, b = self.random_w_and_b()

        for epoch in range(0, num_of_epoch_iterations):
            # tot_loss = 0
            tot_loss = T.zeros((1), dtype=T.float32, requires_grad=True).to(self.device)
            tot_loss.grad = T.zeros(1)
            tot_loss.retain_grad()

            np.random.shuffle(indices)

            for ii in range(len(indices)):
                i = indices[ii]
                x = train_x[i]
                target = train_y[i]
                
                # Считаем вероятность 
                oupt = self.forward(x, w, b)

                # loss по объекту
                loss = (oupt - target).pow(2).sum()

                # Суммарный loss
                tot_loss = loss + tot_loss

            #             tot_loss = tot_loss + T.norm(w, p=2) # l2 reg
            # tot_loss = tot_loss + T.norm(w, p=1) # l1 reg

            tot_loss.backward(retain_graph=True)  # compute gradients

            w.data += -1 * lrn_rate * w.grad.data
            b.data += -1 * lrn_rate * b.grad.data

            w.grad = T.zeros(self.num_of_features)
            b.grad = T.zeros(1)

            if epoch % 10 == 0:
                print("epoch = %4d " % epoch, end="")
                print("   loss = %6.4f" % (tot_loss / 6))

        self.w = w
        self.b = b

        return w, b

    def predict_proba(self, x):

        p = np.empty(len(x), dtype=np.float32)
        i = 0
        for x_i in x:
            p[i] = self.forward(x_i, self.w, self.b)
            i = i + 1

        return p

    def predict(self, x):

        p = np.empty(len(x), dtype=np.float32)
        i = 0
        for x_i in x:
            p[i] = self.forward(x_i, self.w, self.b)
            i = i + 1

        i = 0
        for p_i in p:
            if p_i >= 0.5:
                p[i] = 1
            else:
                p[i] = 0
            i += 1
        return p