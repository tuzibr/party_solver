import time
class Random:
    def __init__(self, a=314159269, c=453806245, m=2 ** 31, method='lcg'):
        # 使用当前时间的作为随机种子
        self.seed = time.time()
        self.x = self.seed
        self.a = a
        self.c = c
        self.m = m
        self.method = method

    def set_seed(self, seed):
        # 设置种子
        self.seed = seed
        self.x = seed

    def set_method(self, method):
        # 设置随机数生成方法，可选为'lcg'和'book_method'
        if method in ['lcg', 'book_method']:
            self.method = method
        else:
            raise ValueError("Method must be 'lcg' or 'book_method'")

    def generate(self, n=1, lb=0.0, ub=1.0):
        """
        生成随机数。
        参数:
        n (int): 生成随机数的数量。
        lb (float): 随机数的下界。
        ub (float): 随机数的上界。
        返回:
        list: 生成的随机数列表。
        """
        if self.method == 'lcg':
            return self._generate_lcg(n, lb, ub)
        elif self.method == 'book_method':
            return self._generate_book_method(n, lb, ub)

    def _generate_lcg(self, n, lb, ub):
        # 使用线性同余生成器生成随机数
        y = []
        for _ in range(n):
            self.x = (self.a * self.x + self.c) % self.m
            rand_num = self.x / self.m
            scaled_num = lb + (ub - lb) * rand_num
            y.append(scaled_num)
        return y

    def _generate_book_method(self, n, lb, ub):
        # 使用书中描述的方法生成随机数
        r1 = 2 ** 35
        r2 = 2 ** 36
        r3 = 2 ** 37
        r = 2657863
        y = []
        for i in range(n + 50):
            if r % 2 == 0:
                r = r + 1
            while r > r1:
                r = r - r1

            r = 5 * r
            if r >= r3:
                r = r - r3
            if r3 > r >= r2:
                r = r - r2
            if r2 > r >= r1:
                r = r - r1
            if r1 > r:
                r = r

            if i >= 50:
                q = r / r1
                scaled_num = lb + (ub - lb) * q
                y.append(scaled_num)
        return y


if __name__ == '__main__':
    # 测试
    rng = Random()
    rng.set_method('lcg')
    y = rng.generate(10, lb=10, ub=20)
    print(y)
    print(sum(y)/len(y))

    rng.set_method('book_method')
    y = rng.generate(10, lb=10, ub=20)
    print(y)
    print(sum(y)/len(y))