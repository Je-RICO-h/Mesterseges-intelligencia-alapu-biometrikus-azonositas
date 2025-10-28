# Mesterséges intelligencia alapú biometrikus azonosítás
## Szerzők: Pál Erik, Lakatos Róbert, Prof. Dr. Hajdu András

A digitális korszakban az adatvagyon védelme az egyik legkritikusabb biztonsági kihívássá vált. Ezt a feladatot tovább nehezíti a felhasználók kényelmi igénye, amely gyakran elégtelen védelmi mechanizmusok alkalmazásához vezet. A hagyományos, felhasználói beavatkozást igénylő hitelesítési eljárások – például a jelszavas vagy kétfaktoros azonosítás – már nem képesek megfelelően kezelni a jogosulatlan hozzáférések növekvő kockázatát. Ennek következtében elengedhetetlenné vált olyan, passzív és folyamatos azonosítást biztosító megoldások alkalmazása, amelyek nagy pontossággal képesek megerősíteni a felhasználó jelenlétét.

A Dolgozatunkban egy XGBoost alapú gépi tanulási módszert mutatunk be, amely kizárólag a felhasználó gépelési dinamikájából következtet a személyazonosságra. A javasolt megközelítés célja az eszköz előtt ülő személy folyamatos azonosítása, alacsony dimenziójú jellemzők – például a billentyűleütés-tartási idők, a billentyűátmeneti idők, a gépelési sebesség és a hibás leütések aránya – elemzésén keresztül. Megoldásunkat összehasonlítottuk egy, a szakirodalomban elterjedt Long Short-Term Memory (LSTM) mélytanuló modellen alapuló megközelítéssel  [1] [2]. Az általunk fejlesztett modell a benchmark adathalmazon 99% pontosságot ért el, amely 1,2%-os javulást jelentett az LSTM-alapú módszerhez képest. Emellett az XGBoost modell alacsony erőforrás-igénye lehetővé teszi a lokális, valós idejű futtatást: egy Intel i7-7700K és i3-10100 processzoron mindössze 5% CPU-terheltség mellett 63 ms válaszidőt értünk el. A dolgozatban formalizáltuk a gépelési dinamikát, mint időfüggő sztochasztikus folyamatot, amely statisztikai alapot teremtett a jelenség elemzéséhez. Továbbá, a spektrális analízis révén formalizáltuk a felhasználóra jellemző idő-ritmus komponenseket.

A végső összehasonlítás érdekében összegyűjtöttük a legfontosabb paramétereket az alábbi táblázatba, ahol kiegészítettük képzési, valamint a modell méretének tulajdonságaival is. 

<img width="582" height="454" alt="image" src="https://github.com/user-attachments/assets/30a87bfb-5487-471d-b651-bca115ea8755" />

A személyi számítógépes futtatás mellett implementációnk kompatibilis az ipari NVIDIA Morpheus SDK keretrendszerrel is, ami növeli a rendszer rugalmasságát és integrálhatóságát. Megoldásunk így egyszerre biztosítja a magánszféra védelmét és a háttérben zajló, észrevétlen működést, a felhasználói kényelem feláldozása nélkül.

Az általunk javasolt technológia egy új, proaktív biztonsági réteg bevezetését teszi lehetővé, amely kiválthatja vagy megerősítheti a hagyományos hitelesítési formákat. Ez a megközelítés utat nyit a jövőbeli önvezérlő biztonsági rendszerek felé, amelyek képesek azonnali védelmi intézkedések – például az eszköz zárolása – végrehajtására jogosulatlan beavatkozás gyanúja esetén.

**Megoldásunk nyílt forráskódú. A tudományos reprodukálhatóság és az elismerés biztosítása érdekében a felhasználás és továbbfejlesztés megfelelő hivatkozáshoz (referenciához) kötött.**

[1] 	BiDAlab, „TypeNet GitHub Repository,” [Online]. Available: https://github.com/BiDAlab/TypeNet. [Hozzáférés dátuma: 20 Október 2025].

[2] 	A. Alejandro, M. Aythami, M. J. V., V.-R. Ruben és F. Julian, „TypeNet: Deep Learning Keystroke Biometrics,” IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021. 

# A program működése

A megoldás teszteléséhez biztosítjuk a tanulmányunkban használt adatkészletet, továbbá ismertetjük a fontosabb elemeket:

### Data
Ez a mappa tartalmazza az általunk gyűjtött és készített adatokat, amellyel kipróbálható a megoldás, továbbá ide kerülnek mentésre az újonnan gyűjtött adatok is.

### Data_Processed
Ez a mappa tartalmazza, a "Processing" Mappában lévő .ipynb fájl által feldolgozott nyers adatokat. Ezek az adatok már rendezettek, augmentáltak és kiegészítettek minden tulajdonsággal

### Inference
Az inferenciához szükséges modellbetöltő és jósló osztályok, valamint függvények összessége. Modellenként csoportosítva.

### Logging
Az egyedileg készített keylogger fájljait és működéséhez szükséges kódot tartalmazza, itt is Modellenként csoportosítva, azonban adatgyűjtéshez ajánlott a "Keyboard_Logger_XgBoost.py" szkriptet használni

### Model_Training
A modellek képzéséhez/implementálásához használt programok, szintén modellenként csoportosítva

### Processing
A nyers adatok augmentálásához, feldolgozásához szükséges eljárásokat tartalmazza, a kimenete a "Data_Processed" mappába kerül.

### Trained_Model
Az inferenciára kész, betanult modelleket és a hozzájuk szükséges egyéb fájlokat tartalmazza (OHE features, Label encoder stb.), modellenként csoportosítva.


### Fő fájlok
A programot a két "Train_data_collect_(Model).py fájlokbol kezeljük, annak függvényében melyik modell-t szeretnénk használni inferenciára.
- A model_path változó tartalmazza az elérési utakat a modellekhez, ezt alap esetben nem kell módosítani.
- A threshold-al a gyűjtött mintaszámot tudjuk változtatni
- a file_path paraméterrel az adatok mentési mappáját tudjuk megváltoztatni
- inference_mode - True értékkel **inferencia** módban fog futni a program a megfelelő modell-t betöltve, False esetén **adatgyűjtési** módba
- A program futtatásakor nincs más dolgunk, mint megadjuk a kívánt label nevét amikor a program bekéri, majd az enterrel indítható a program és már csak gépelni szükséges. A program végzi a többi feladatot a háttérben.

### Továbbiak
- A környezet futtatásához szükséges könyvtárak és conda környezet megtalálható az environment.yml és Requirements.txt fájlokban.

#Felhasználás és Hivatkozás
Örömmel vesszük, ha kódunkat és módszertanunkat felhasználod tudományos kutatásokban, kereskedelmi projektekben vagy más nyílt forráskódú kezdeményezésekben!

Kérjük, figyelembe venni, hogy munkánk integritása érdekében minden, a projektből származó kód, adatok vagy eredmény felhasználása esetén a következő formátum szerint adj megfelelő hivatkozást (referenciát) az eredeti forrásra.
