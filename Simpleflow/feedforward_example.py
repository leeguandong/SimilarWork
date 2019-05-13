import Simpleflow.simpleflow as sf

with sf.Graph().as_default():
    a = sf.constant(1.0, name='a')
    b = sf.constant(2.0, name='b')
    result = a + b

    # Create a session to run the graph
    with sf.Session() as sess:
        print(sess.run(result))

        # Create a graph
with sf.Graph().as_default():
    w = sf.constant([[1, 2, 3], [3, 4, 5]], name='w')
    x = sf.constant([[9, 8], [7, 6], [10, 11]], name='x')
    b = sf.constant(1.0, 'b')
    result = sf.matmul(w, x) + b

    # Create a session to compute
    with sf.Session() as sess:
        print(sess.run(result))
