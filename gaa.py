def funn(v):
    global c
    global normalize
    global myK
    sig = v[0]
    if sig <= 0:
        sig = 0.0001
    K = int(v[1])
    if K <= 0:
        K = 1
    elif K > myK:
        K = myK
    nca = NeighborhoodComponentsAnalysis(random_state=42, sigma=sig)
    nca.fit(X_train, y_train)

    # df1 = pd.DataFrame(nca.transform(X_train))
    # df1.to_csv("X_train_embed_{}_nca.csv".format(db_name), index=False, header=False)
    # df2 = pd.DataFrame(nca.transform(X_test))
    # df2.to_csv("X_test_embed_{}_nca.csv".format(db_name), index=False, header=False)

    knn = KNeighborsClassifier(n_neighbors=K)


    knn.fit(nca.transform(X_train), y_train)

    return 1 - knn.score(nca.transform(X_test), y_test)
