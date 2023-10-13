
1. Sammanfatta samtliga föreläsningar, leta efter frågor som kan ställas.
2. Gå igenom labbarna och förstå vad som händer och varför.

# Sammanfattning av föreläsningar

## KNN 

#### **KNN Klassifiering.**

**Definition :** 
1. Classify a new example x by finding the training example (xi,yi) that is  
nearest to x
2. k-NN is a simple algorithm that stores all available samples and classifies new  
samples based on a similarity measure.

**Likhetsfunktioner:** 
Euclidian, Manhattan, Minowski. Är olika sätt att mäta avståndet mellan två datapunkter.

**Hur den fungerar:**
Vi lägger okänt data baserat på dens distans vilket är dens egenskaper som vi skickar till en likhetsfunktion.
Vi kollar om K närmsta grannar (K är antalet grannar vi kollar på för att bestämma vad den okända datat är.) 
Vi räknar hur många klasser det finns och hur många som tillhör klasserna. Den klass som vinner tilldelar vi den okända datat.

**Hur väljer man ett bra K**
En tumregel är att man sätter K < sqrt(n), där n är antalet datapunkter i hela datasettet. Man vill ha ett ojämt k värde också.
_____________________________________________________________________________

#### KNN Regression: 

**Vad är regression?** 
Regression inom maskininlärning är en statistisk metod för att modellera sambandet mellan en beroende variabel och en eller flera oberoende variabler.

k-NN (k-närmaste grannar) regression är en typ av instansbaserad inlärning eller icke-parametrisk regressionsalgoritm. Den använder träningsdata för att göra förutsägelser och skapar inte en modell för att användas med nya datapunkter. Algoritmen gör inga antaganden om träningsdatan.

I k-NN regression förutsäger vi värdet på en beroende variabel för en datapunkt baserat på medelvärdet eller genomsnittet av målvärdena för dess k närmaste grannar i träningsdatan.

**Hur fungerar det?** 
Det är nästan samma sak förutom att vi räknar ut medelvärdet av de närmaste grannarnas värde. Tex i boston datasettet där vi vill veta huspriset då tar vi medelvärdet av de närmsta grannarna och sätter det till det okända datat.






#### Fördelar / Nackdelar
###### Fördelar:

• Enkelt att förstå och implementera: Algoritmen är lätt att förstå och implementera, vilket gör den tillgänglig för både praktiker och forskare.

• Inga antaganden om datadistribution: Detta gör den lämplig för en bred variation av dataset, inklusive de med komplexa eller icke-linjära relationer mellan egenskaper och målvariabler.

• Hanterar brusig data bra: kNN kan hantera brusig data väl och är mindre känslig för utliggare och extrema värden i datamängden.

• Mångsidig: kNN kan användas för både regressions- och klassificeringsproblem samt kan hantera både kontinuerliga och kategoriska målvariabler.

• Används för online-lärande: kNN kan användas för online-lärande, vilket innebär att det kan uppdateras stegvis när ny data blir tillgänglig. Detta kan ge exakta resultat i realtid.

• Fungerar bra med små dataset: Till skillnad från vissa andra algoritmer som kräver en stor mängd data för att fungera optimalt, kan kNN fortfarande producera bra resultat med små dataset, så länge data är representativt för problemområdet.

###### Nackdelar:
• Beräkningskostnad: Vid förutsägelser behöver kNN-algoritmen beräkna avståndet mellan alla datapunkter i den befintliga datamängden och den nya datamängden. Detta leder till att beräkningskostnaderna ökar i takt med att datamängden blir större. Därför kan den vara ineffektiv vid användning med stora datamängder.

• Högt minnesanvändning: kNN kräver mycket minne för att lagra hela träningsdatamängden, vilket kan vara ett problem för mycket stora datamängder.

• Känslighet för hyperparametrar: Prestandan för kNN beror starkt på valet av hyperparametern K, som avgör antalet närmaste grannar som används för att göra förutsägelser. Att välja fel värde för K kan leda till överanpassning eller underanpassning.

• Känslighet för irrelevanta egenskaper: kNN är känslig för irrelevanta egenskaper eftersom de kan ha en stor påverkan på avståndsmåttet som används för att identifiera de närmaste grannarna. Detta kan leda till dålig prestanda om egenskaperna inte preprocessas noggrant.

• Icke-parametrisk natur: Till skillnad från andra regressionsalgoritmer ger kNN regression inte en modell som kan användas för att göra förutsägelser för nya datapunkter. Varje gång beräknar den avståndet och ger sedan resultatet. Detta kan göra det svårare att tolka resultaten och förstå relationerna mellan egenskaperna och målvariablerna.

• Inte lämplig för stora datamängder med många funktioner: kNN kan bli beräkningsmässigt olämplig för datamängder med många funktioner och datapunkter. I dessa fall kan du använda algoritmer som multipel regression eller polynomisk regression.