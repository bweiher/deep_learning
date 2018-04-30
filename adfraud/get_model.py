
def get_model(lr=0.001, decay=0.0):
    ip = Input(shape=[1], name="ip")
    app = Input(shape=[1], name="app")
    device = Input(shape=[1], name="device")
    os = Input(shape=[1], name="os")
    channel = Input(shape=[1], name="channel")
    clicktime = Input(shape=[1], name="clicktime")

    emb_ip = Embedding(MAX_IP, 64)(ip)
    emb_device = Embedding(MAX_DEVICE, 16)(device)
    emb_os= Embedding(MAX_OS, 16)(os)
    emb_app = Embedding(MAX_APP, 16)(app)
    emb_channel = Embedding(MAX_CHANNEL, 8)(channel)
    emb_time = Embedding(MAX_TIME, 32)(clicktime)

    main = concatenate([Flatten()(emb_ip), 
                        Flatten()(emb_device), 
                        Flatten()(emb_os),
                        Flatten()(emb_app),
                        Flatten()(emb_channel), 
                        Flatten()(emb_time)])
    main = Dense(128,kernel_initializer='normal', activation="tanh")(main)
    main = Dropout(0.2)(main)
    main = Dense(64,kernel_initializer='normal', activation="tanh")(main)
    main = Dropout(0.2)(main)    
    main = Dense(32,kernel_initializer='normal', activation="relu")(main)
    output = Dense(1,activation="sigmoid") (main)
    #model
    model = Model([ip, app, device, os, channel, clicktime], output)
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="binary_crossentropy", 
                  optimizer=optimizer)
    return model
