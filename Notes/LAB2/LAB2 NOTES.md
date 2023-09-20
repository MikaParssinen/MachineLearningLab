Beslutsträd är ett verktyg som tar beslut i en form av ett träd. Den räknar ut möjliga konsekvenserna för varje beslut. Det är ett sätt att visualisera arbetsättet av en förutsägelse algoritm. 
Exempel :
![[Screenshot 2023-09-20 at 16.22.51.png]]

Besluten representeras i flera olika verbala tillstånd. Dessa tillstånd är baserat på beslut som har tagits högre upp eller av beslut som kommer tas längre ner. Ett beslutsträd har två olika funktioner:
1. Data kompression, ett sätt att visa och representera data
2. Förutsäga, kan användas att klassificera nya objekt

**TDIDT**, Top-Down Induction of Decision Tree
En algoritm som har funnits sen 1990-talet, två av de bästa klassificerings
systemen **ID3** och **C4.5** har blivit formade utan denna algoritm.

Beslutsträd generas genom upprepad splittring av värden av attributen, denna process kallas för rekursiv partitionering. 
1. IF, alla instanser i tränings settet till hör samma klass
2. THEN, returnera värdet av klassen
3. ELSE, (a) Välj attribut A för att splitta*. (b) Sortera instanserna i tränings settet till delmängder en för varje värde av attribut A. (c) Returnera ett träd med en gren för vare icke tom delmängd, varje gren har ett underträd som avkom, eller en klass värde producerat genom att applicera algoritmen rekursivt.
*Välj aldrig ett attribut två gånger*

**Förkunskaper av TDIDT**:
**Tillräcklighetsvillkoret** måste hållas av träningsuppsättningen innan algoritmen tillämpas.
Villkoret säger att: Inga två instanser med samma värde av alla attribute tillhör olika klasser.
Detta är ett sätt att försäkra sig att tränings datat är konsekvent.

**Entropi-baserat lärande**:
1. Väljande av attribut
2. Entropi 
3. Lens Dataset
4. Användandet av entropi i väljande av attribut
### **1**. Väljande av attribut
Det finns tre vanliga sätt att välja attribut utan förkunskap. 

• **Take First** - För varje gren ta de attribut i den ordningen dom uppstår i data settet

• **Take Last** - För varje gren ta de attribut i omvänd ordning som dom uppstår i.

• **Take Randomly** - Gör randomiserade beslut med lika mycket sannolikhet på att varje attribut blir taget.

Dessa metoder minskar värdet av Entropi och maximerar den informationen vi får.
.
### **2**. Entropi

•Entropi är en informations-teoretisk mått på ösäkerhet som finns i träningsdata, pågrund av att det finns mer än en möjlig klassificering.
•Om det finns **K** klasser kan vi beteckna andelen instanser med klassificering **i** med **p_i** för i = 1 till K.
• Värdet av **p_i** är nummret av instanser av klass **I** dividerat med totala numret instanser vilket kommer vara en siffra mellan 0 och 1.
• Entropin tar sit maximala värde när instanser är lika fördelade mellan de **K** olika klasser.

För resten kolla canvas efter slidsen kNN(with Regression), Logi..... osv den har ett bra exempel med Lens Dataset. Lens dataset har 3 klasser och 4 


