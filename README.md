# Étude sur les offres d’hébergement de Booking.com

## Contexte du projet
Ce projet a été réalisé dans le cadre du **Master Informatique - parcours MIND ex-DAC** à **Sorbonne Université**.  
L’objectif principal est d’analyser les dynamiques du marché hôtelier en exploitant des données issues de **Booking.com** et d’autres sources (Numbeo, Worldometer, World Population Review, SimpleMaps).  

Nous avons étudié les relations entre les prix, les scores, la localisation des hôtels, le niveau de vie local et les données démographiques, afin d’apporter des insights utiles pour l’industrie hôtelière, la recherche académique et les décideurs.

---

##  Contenu du projet
Le dépôt contient plusieurs modules organisés en notebooks et scripts Python :

1. **Collecte des données**  
   - Web scraping avec *BeautifulSoup* et automatisation en Python.  
   - Extraction depuis Booking.com (prix, scores, localisation, disponibilité), Numbeo (coût de la vie), et autres sources démographiques.  
   - Sauvegarde des données au format **CSV**.

2. **Nettoyage et prétraitement**  
   - Formatage, correction des valeurs incohérentes et suppression des doublons.  
   - Fusion des données hôtelières avec les données urbaines et démographiques.  

3. **Analyses et visualisations**  
   - Distribution des hôtels par continent, prix et scores.  
   - Étude des politiques d’annulation, services de petit-déjeuner, et distances au centre-ville.  
   - Corrélations entre variables hôtelières et environnementales.  
   - Analyse en Composantes Principales (ACP).  

4. **Modélisation prédictive**  
   - Régression (régression linéaire, Random Forest Regressor) pour prédire le **score moyen des hôtels**.  
   - Classification supervisée (KNN, Random Forest) pour prédire le **continent**, la **région**, le **pays** et la **ville**.  

5. **Tableau de bord interactif**  
   - Construit avec **Dash & Plotly**.  
   - Permet d’explorer les distributions et relations entre variables (scores, prix, services, localisation).  
   - Visualisations interactives mises à jour dynamiquement selon les sélections de l’utilisateur.

----

## Résultats principaux
- **Corrélations** : les hôtels avec des prix modérés obtiennent souvent de meilleurs scores.  
- **Modélisation** :  
  - Prédiction du score des hôtels avec **Random Forest Regressor (R² ≈ 0.74)**.  
  - Classification par continent, région, pays, et ville avec une précision proche de **99 %**.  
- **Dashboard** : exploration interactive des tendances hôtelières mondiales.  

---

## Auteurs
- **Yacine  Chettab**  
- **Carine Moubarak**  
Projet supervisé par **Laure Soulier**  
