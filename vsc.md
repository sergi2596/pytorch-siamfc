# How to use the Vienna Scientific Cluster GPU Server

1. Create Python environment:

    load python3 module
    ```
    module load python/3.6
    ```
    create your own virtual environment
    ```
    python3 -m venv env
    ```
    active environment
    ```
    source my-env/bin/activate
    ```
    install your required packages using pip3 (e.g. torch, tensorflow-gpu, ...)

1. Create trainings script

    As an simple example we use mnist here
    - When using Tensorflow e.g train_mnist.py: 
        ```
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)
        ````
    - When using pytorch e.g. download file from [https://pytorch.org/tutorials/_downloads/cifar10_tutorial.py]
        and insert 
        ```
        import matplotlib
        matplotlib.use('Agg)
        ```
        at line 85 before "import matplotlib.pyplot as plt" because we have no X server


1. Create a bash file (e.g. start_training.sh) which starts your job

    
    ``` bash 
    #!/bin/bash
    #SBATCH -J my_tensorflow
    #SBATCH -N 1
    #SBATCH --partition gpu_gtx1080single
    #SBATCH --qos gpu_gtx1080single
    #SBATCH --gres gpu:1
    #SBATCH --mail-type=ALL    # first have to state the type of event to occur  (BEGIN, END, FAIL, REQUEUE, ALL)
    #SBATCH --mail-user=<email@address.at>   # and then your email address

    module purge
    module load gcc/6.4 python/3.6 cuda/9.0.176

    python [my_train_script.py]
    ```

1. Submit your job with command
    ```
    sbatch ./start_training.sh
    ```
1. Check Status of Job
    ```
    squeue -u [username]
    ```
    in the ST column is the state:
    
    R... running

    PD ... pending

    CD ... Completed

    CG ... Completing (job is not really canceled, but you cannot do anything about it. or?)

1. Output of the job will be written to slum-[JOBID].out


## misc
- cancel job with
    ```
    scancel [JOBID]
    ```
