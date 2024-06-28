import numpy as np


def identity_function(x: np.ndarray | np.number) -> np.ndarray | np.number:
    return x


def sigmoid(x):
    """シグモイド関数
    本の実装ではオーバーフローしてしまうため、以下のサイトを参考に修正。
    http://www.kamishima.net/mlmpyja/lr/sigmoid.html

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    # xをオーバーフローしない範囲に補正
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)

    # シグモイド関数
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """5章で学ぶ関数。誤差逆伝播法を使う際に必要。"""
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """交差エントロピー誤差の算出

    Args:
        y (numpy.ndarray): ニューラルネットワークの出力
        t (numpy.ndarray): 正解のラベル

    Returns:
        float: 交差エントロピー誤差
    """

    # データ1つ場合は形状を整形
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 誤差を算出してバッチ数で正規化
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def numerical_gradient(f, x):
    """勾配の算出

    Args:
        f (function): 損失関数
        x (numpy.ndarray): 勾配を調べたい重みパラメーターの配列

    Returns:
        numpy.ndarray: 勾配
    """
    h = 1e-4
    grad = np.zeros_like(x)

    # np.nditerで多次元配列の要素を列挙
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index  # it.multi_indexは列挙中の要素番号
        tmp_val = x[idx]  # 元の値を保存

        # f(x + h)の算出
        x[idx] = tmp_val + h
        fxh1 = f()

        # f(x - h)の算出
        x[idx] = tmp_val - h
        fxh2 = f()

        # 勾配を算出
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を戻す
        it.iternext()

    return grad
