# Pregunta 2: A quins tipus de peça de roba pertanyen els tres veïns més propers de la primera imatge del test set 
    # segons el KNN si aquest s’inicialitza amb totes les imatges del train set? Nota: l’ordre que apareixen els tipus 
    # de peça de roba no és rellevant.
    knn = KNN(train_imgs_grayscale, train_class_labels)
    knn.get_k_neighbours(test_imgs_grayscale[1], 3)
    print(knn.self.neighbors)