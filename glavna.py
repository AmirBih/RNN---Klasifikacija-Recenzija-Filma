#Ucitavanje potrebnih kofera alata
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

#Ucitaj indekse rijeci od IMDB niza podataka
indeks_rijeci = imdb.get_word_index()
reverz_indeks_rijeci = {value: key for key, value in indeks_rijeci.items()}

#Ucitaj Istrenirani Model i postavi RELU aktivaciju
model = load_model("SimpleRNN_imdb.h5")

#Korak 2 - Stvaranje Funkkcija za Dekodiranje Revjua i Predprocesovanje Teksta

def kodiranje_revjua(kodiran_revju):
    return " ".join([reverz_indeks_rijeci.get(i-3,"?") for i in kodiran_revju])

#Funkcija za Sredjivanje Prvobitnog Teksta
def sredi_tekst(tekst):
    rijeci = tekst.lower().split()
    kodiran_revju = [indeks_rijeci.get(rijec, 2) +3 for rijec in rijeci]
    padiran_revju = sequence.pad_sequences([kodiran_revju], maxlen=500)
    return padiran_revju

#Korak 3 - Stvaranje Funkcije za Obavljanje Predviđanja
def predvidi_znacenje(revju):
    sredjen_unos = sredi_tekst(revju)
    predvidjanje = model.predict(sredjen_unos)
    znacenje = "Pozitivno" if predvidjanje[0][0] >0.5 else "Negativno"
    
    return znacenje, predvidjanje[0][0]

#Streamlit aplikacija
import streamlit as st
st.title("Analiza Sentimentalnosti IMDB Revjua o Filmovima - Pozitivno ili Negativno")
st.write("Unesi Mišljenje/Recenziju o Nekom Filmu (Na Engleskom).")

#Korisnicki Unos
unos_korisnika = st.text_area("Revju o Nekom Filmu")

if st.button("Klasifikuj"):
    sređeni_tekst = sredi_tekst(unos_korisnika)
    
    #Stvaranje Predikcije
    predvidjanje = model.predict(sređeni_tekst)
    sentimentalnost = "Pozitivno" if predvidjanje[0][0] >0.5 else "Negativno"
    
    #Prikazi taj rezultat
    st.write(f"Sentimentalnost: {sentimentalnost}")
    st.write(f"Rezultat Klasifikacije: {round(predvidjanje[0][0],2)}")
else:
        st.write("Molimo unesite mišnjenje o nekom filmu.")
    




