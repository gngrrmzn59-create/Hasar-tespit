import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from fpdf import FPDF

st.set_page_config(page_title="Otobüs Hasar Takip", layout="wide")

# Klasör yapısı
if not os.path.exists("data"): os.makedirs("data")

def hizala_ve_isaretle(img_eski, img_yeni):
    # (Önceki hizalama mantığı buraya dahil edilir)
    # Fark tespiti ve kontur (kare) çizimi:
    g1 = cv2.cvtColor(img_eski, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_yeni, cv2.COLOR_BGR2GRAY)
    fark = cv2.absdiff(g1, g2)
    _, esik = cv2.threshold(fark, 30, 255, cv2.THRESH_BINARY)
    
    # Küçük gürültüleri temizle
    esik = cv2.dilate(esik, None, iterations=2)
    konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hasar_sayisi = 0
    img_sonuc = img_yeni.copy()
    
    for c in konturlar:
        if cv2.contourArea(c) < 500: # Çok küçük değişimleri görmezden gel
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_sonuc, (x, y), (x + w, y + h), (0, 0, 255), 3)
        hasar_sayisi += 1
        
    return img_sonuc, hasar_sayisi

# Arayüz kodları... (Önceki akışla aynı)
