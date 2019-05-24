import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw
import csv
import face_recognition
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import keras
from tkinter import filedialog
from tkinter import*
import collections

def obidjiDirektorijume():

    rezultat = []
    ids = []
    bazni_direktorijum = os.path.dirname(os.path.abspath(__file__))
    putanja_slika = os.path.join(bazni_direktorijum, "feret")

    # Obilazak svih direktorijuma u kojima se nalaze lica
    for root, direktorijum, fajlovi in os.walk(putanja_slika):
        for fajl in fajlovi:
            if fajl.endswith("jpg") or fajl.endswith("Jpg") or fajl.endswith("pgm") or fajl.endswith("ppm"):
                putanja = os.path.join(root, fajl)
                id = os.path.basename(os.path.dirname(putanja)).replace(" ", "-").lower()

                slika = PIL.Image.open(putanja)
                # image = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

                # Skaliramo sve slike da budu dimenzije 290x320
                obradjena = slika.resize((290, 320))
                obradjena_direktorijum = os.path.join(bazni_direktorijum, "za_obradu.jpg")
                obradjena.save(obradjena_direktorijum)

                slika = face_recognition.load_image_file(obradjena_direktorijum)

                # Izdvajanje svih lica na slici
                listaLica = face_recognition.face_landmarks(slika)

                if(listaLica != []):
                    for lice in listaLica:
                        ids.append(id)
                        lice = collections.OrderedDict(sorted(lice.items(), key=lambda t: t[0]))
                        for crteLica in lice.keys():
                            if (crteLica != "right_eye" and crteLica != "left_eye" and crteLica != "top_lip" and
                                    crteLica != "bottom_lip"):
                                rezultat.append(duzinaISirina(lice[crteLica]))
                            else:
                                if (crteLica != "bottom_lip"):
                                    rezultat.append(duzinaISirinaUstaIOciju(lice[crteLica]))

                        rezultat.append(euklidskoRastojanje(lice["left_eye"][len(lice["left_eye"]) - 1],
                                                            lice["right_eye"][0]))
                        rezultat.append(euklidskoRastojanje(lice["left_eyebrow"][len(lice["left_eyebrow"]) - 1],
                                                            lice["right_eyebrow"][0]))


    np.savetxt("podaci_x.csv", rezultat)
    np.savetxt("podaci_y.txt", ids, fmt="%s")


def slikaZaTest(leviOkvir):
    f = open("putanja.txt", "r+")
    putanja = f.readline()
    f.close()
    slika = PIL.Image.open(putanja)

    podaciX = []
    podaciY = []

    with open('podaci_x.csv', 'r') as f1:
        reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            podaciX.append(row[0])

    f1.close()

    with open('podaci_y.txt') as f2:
        for linija in f2:
            podaciY.append(linija.strip())

    f2.close()

    y_trening = np.array(podaciY)
    x_trening = np.array(podaciX).reshape(len(podaciY), 10)

    obradjena = slika.resize((290, 320))
    bazni_direktorijum = os.path.dirname(os.path.abspath(__file__))
    obradjena_direktorijum = os.path.join(bazni_direktorijum, "za_obradu.jpg")
    obradjena.save(obradjena_direktorijum)

    slika = face_recognition.load_image_file(obradjena_direktorijum)
    pil_slika = PIL.Image.fromarray(slika)
    d = PIL.ImageDraw.Draw(pil_slika)

    listaLica = face_recognition.face_landmarks(slika)

    rezultat = obradiSliku(obradjena_direktorijum)
    #print(rezultat)

    if (rezultat != None):
        x_test = np.array(rezultat).reshape(1, 10)

        for widget in leviOkvir.winfo_children():
            widget.destroy()
        for lice in listaLica:
            for crteLica in lice.keys():
                d.line(lice[crteLica], width=5)

        tkpi = PIL.ImageTk.PhotoImage(pil_slika)
        mestoZaSluku = Label(leviOkvir, image=tkpi, width=400, height=350)
        mestoZaSluku.image = tkpi
        mestoZaSluku.pack(padx=20)
        treniraj2(x_trening, y_trening, x_test, leviOkvir)
    else:
        greska = Label(leviOkvir, text="Ne postoji lice na slici")
        greska.pack(pady = 5)



def obidjiDirektorijumeZaTest(leviOkvir):

    for widget in leviOkvir.winfo_children():
        widget.destroy()
    podaciX = []
    podaciY = []


    with open('podaci_x.csv', 'r') as f1:
        reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            podaciX.append(row[0])

    f1.close()

    with open('podaci_y.txt') as f2:
        for linija in f2:
            podaciY.append(int(linija.strip()))

    f2.close()

    Y = np.array(podaciY)
    X = np.array(podaciX).reshape(len(podaciY), 10)

    x_trening, x_test, y_trening, y_test = train_test_split(X, Y, test_size=0.2)

    trenirajZaTest(x_trening, y_trening, x_test, y_test, leviOkvir)


def obradiSliku(putanja):
    slika = face_recognition.load_image_file(putanja)
    listaLica = face_recognition.face_landmarks(slika)
    rezultat = []
    for lice in listaLica:
        lice = collections.OrderedDict(sorted(lice.items(), key=lambda t: t[0]))
        for crteLica in lice.keys():
            if (crteLica != "right_eye" and crteLica != "left_eye" and crteLica != "top_lip" and
                    crteLica != "bottom_lip"):
                rezultat.append(duzinaISirina(lice[crteLica]))
            else:
                if (crteLica != "bottom_lip"):
                    rezultat.append(duzinaISirinaUstaIOciju(lice[crteLica]))

        rezultat.append(euklidskoRastojanje(lice["left_eye"][len(lice["left_eye"]) - 1],
                                            lice["right_eye"][0]))
        rezultat.append(euklidskoRastojanje(lice["left_eyebrow"][len(lice["left_eyebrow"]) - 1],
                                            lice["right_eyebrow"][0]))

        return rezultat


def prikaziMatricuKonfuzije(cm):

    plt.figure()
    dimenzija = len(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrica kofuzije')
    plt.colorbar()
    plt.xticks(range(dimenzija), range(1, dimenzija+1))
    plt.yticks(range(dimenzija), range(1, dimenzija+1))


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('Tacne vrednosti')
    plt.xlabel('Predvidjene vrednosti')
    plt.tight_layout()

    plt.show()
    plt.close()

def trenirajZaTest(x_trening, y_trening, x_test, y_test, leviOkvir):
    # Kreiranje neuronske mreze
    model = Sequential([
        Dense(20, input_dim=10),
        Activation('relu'),
        Dense(50),
        Activation('relu'),
        Dense(70),
        Activation('relu'),
        Dense(15),
        Activation('softmax'),
    ])

    # Kompajliranje mreze
    model.compile(optimizer= 'adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Konvertovanje izlaza u kategoricki podatak
    treningLabela = keras.utils.to_categorical(y_trening)

    # Ucenje mreze
    model.fit(x_trening, treningLabela, epochs=1000)

    # Izracunavanje predvidjenih vrednosti izlaza za test i trening skup
    y_pred_trening = model.predict_classes(x_trening)
    y_pred_test = model.predict_classes(x_test)

    print(y_test)
    print(len(y_test))
    print(y_pred_test)

    brojPogodjenihTrening = 0
    for i in range(0, len(y_trening)):
        if(y_trening[i] == y_pred_trening[i]):
            brojPogodjenihTrening = brojPogodjenihTrening + 1

    brojPogodjenihTest = 0
    for i in range(0, len(y_test)):
        if (y_test[i] == y_pred_test[i]):
            brojPogodjenihTest = brojPogodjenihTest + 1


    preciznostNaTreningSkupu = brojPogodjenihTrening*100.0/ len(y_trening)
    preciznostNaTestSkupu = brojPogodjenihTest * 100.0 / len(y_test)

    labelaRezultat1 = Label(leviOkvir, text="Preciznost na trening skupu je " + str(round(preciznostNaTreningSkupu, 2)))
    labelaRezultat1.pack(padx=20)


    labelaRezultat2 = Label(leviOkvir, text="Preciznost na test skupu je " + str(round(preciznostNaTestSkupu, 2)))

    labelaRezultat2.pack(pady=10, padx=20)

    matricaKonfuzije = confusion_matrix(y_trening, y_pred_trening)
    matricaKonfuzijeTest = confusion_matrix(y_test, y_pred_test)

    dugmeZaMatricu = Button(leviOkvir, text='Prikazi matricu konfuzije', width=20,
                          command=lambda: prikaziMatricuKonfuzije(matricaKonfuzijeTest))
    dugmeZaMatricu.pack(pady=10, padx=20)

def treniraj2(x_trening, y_trening, x_test, leviOkvir):
    # Kreiranje neuronske mreze
    model = Sequential([
        Dense(20, input_dim=10),
        Activation('relu'),
        Dense(50),
        Activation('relu'),
        Dense(70),
        Activation('relu'),
        Dense(15),
        Activation('softmax'),
    ])

    # Kompajliranje mreze
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Konvertovanje izlaza u kategoricki podatak
    treningLabela = keras.utils.to_categorical(y_trening)

    # Ucenje mreze
    model.fit(x_trening, treningLabela, epochs=1000)

    y_pred = model.predict_classes(x_test)


    labelaRezultat = Label(leviOkvir, text="Osoba " + str(y_pred[0]))
    labelaRezultat.pack(pady=10)

    nizVerovatnoca = model.predict_proba(x_test)
    for i in range(0, len(nizVerovatnoca[0])):
        verovatnoca = nizVerovatnoca[0][i] * 100.00
        if (verovatnoca > 10 and verovatnoca < 50):
            labelaVerovatnoca = Label(leviOkvir,
                                      text="Ova osoba " + "lici na osobu " + str(i) + " u procentu od " + str(
                                          round(verovatnoca, 2)) + "%")
            labelaVerovatnoca.pack(pady=5)



def euklidskoRastojanje(tacka1, tacka2):
    return math.sqrt(
        (tacka1[0] - tacka2[0]) * (tacka1[0] - tacka2[0]) + (tacka1[1] - tacka2[1]) * (tacka1[1] - tacka2[1]))


def duzinaISirina(nizTacaka):
    duzinaNiza = len(nizTacaka)
    rastojanje = euklidskoRastojanje(nizTacaka[0], nizTacaka[duzinaNiza - 1])

    return rastojanje


def duzinaISirinaUstaIOciju(nizTacaka):
    duzinaNiza = len(nizTacaka)
    rastojanje = 0
    for i in range(1, duzinaNiza - 1):
        if (nizTacaka[i - 1][0] > nizTacaka[i][0]):
            rastojanje = euklidskoRastojanje(nizTacaka[0], nizTacaka[i - 1])
            break

    return rastojanje


def ucitaj_sliku(leviOkvir):
    for widget in leviOkvir.winfo_children():
        widget.destroy()

    imeSlikeZaObradu = filedialog.askopenfilename(initialdir="/Desktop", title="Select file",
                                                  filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

    f = open("putanja.txt", "w+")
    f.write(imeSlikeZaObradu)
    ucitavanje = PIL.Image.open(imeSlikeZaObradu)
    ucitavanje = ucitavanje.resize((290, 320))
    slika = PIL.ImageTk.PhotoImage(ucitavanje)
    mestoZaSluku = Label(leviOkvir, image=slika, width = 300, height=390)
    mestoZaSluku.image = slika
    mestoZaSluku.pack(padx=20)


def napraviGui():
    root = Tk()
    root.title("Prepoznavanje lica")

    prozor = Frame(master=root, width=400, height=400)
    leviOkvir = Frame(master=prozor, width=300, height=400)
    leviOkvir.pack(side=LEFT)

    desniOkvir = Frame(master=prozor, width=200, height=400)
    desniOkvir.pack(side=RIGHT)

    nadjiSliku = Button(master=desniOkvir, text='Ucitaj sliku', width=20, command=lambda: ucitaj_sliku(leviOkvir))
    nadjiSliku.pack(side=TOP, padx=20)

    treniraj = Button(master=desniOkvir, text='Obradi slike', width=20, command=lambda: obidjiDirektorijume())
    treniraj.pack(side=TOP, padx=20)


    testiraj = Button(master=desniOkvir, text='Testiraj jednu sliku', width=20, command=lambda: slikaZaTest(leviOkvir))
    testiraj.pack(side=TOP, padx=20)

    testirajVise = Button(master=desniOkvir, text='Testiraj vise slika', width=20, command=lambda: obidjiDirektorijumeZaTest(leviOkvir))
    testirajVise.pack(side=TOP, padx=20)


    izlaz = Button(master=desniOkvir, text='Izlaz', width=20, command=lambda: quit())
    izlaz.pack(side=BOTTOM, padx=20, pady=100)
    prozor.pack()
    root.mainloop()

def main():
    napraviGui()



if __name__ == "__main__":
    main()