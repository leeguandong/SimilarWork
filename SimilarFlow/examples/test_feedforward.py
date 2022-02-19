'''
@Time    : 2022/2/17 16:23
@Author  : leeguandon@gmail.com
'''
import similarflow as sf

with sf.Graph().as_default():
    w = sf.Constant([[1, 2, 3], [3, 4, 5]])
    x = sf.Constant([[9, 8], [7, 6], [10, 11]])
    b = sf.Constant(1.0)
    result = sf.add(sf.matmul(w, x), b)
    with sf.Session() as sess:
        print(sess.run(result))
