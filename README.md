# Dokumentacja modelu lepkosprężystego (`miki_fix.py`)

## 1. Cel skryptu
Skrypt `miki_fix.py` zawiera działającą implementację rozszerzenia standardowego modułu wytrzymałościowego (`MomentumBalance`) w bibliotece **PorePy**. Oryginalny model opierał się na prawie Hooke'a (materiał czysto sprężysty). Skrypt wprowadza materiał **lepkosprężysty**, co skutkuje pojawieniem się drugiej składowej przemieszczenia (`u2`), reprezentującej część lepką. Ostateczne przemieszczenie w takim materiale jest zależne od czasu.

## 2. Architektura i Pętla Obliczeniowa
Skrypt ściśle przestrzega architektury biblioteki PorePy, opartej na wzorcu projektowym **Mixin Module** (wielodziedziczenie). Zamiast modyfikować kod źródłowy biblioteki, funkcjonalność dla zmiennej `u2` została podzielona na logiczne bloki (tworzenie zmiennych, równania, prawa konstytutywne, warunki początkowe, strategia rozwiązywania).

Te oddzielne klocki (Mixins) są na samym końcu łączone z bazową klasą `pp.MomentumBalance` w finalnej klasie układu `ViscoelasticMomentumBalance`.

### Najważniejsze elementy mechaniki modelu:
*   **Zmienne**: Poza standardowym przemieszczeniem sprężystym (`u`), model tworzy i rozwiązuje niezależnie pole przemieszczeń lepkich (`u2`) za pomocą zmiennych `displacement2` (w domenie) i `interface_displacement2` (na szczelinach, jeśli występują).
*   **Równania**: Stworzono zestaw równań zachowania pędu (`momentum_balance_equation2`) oraz równania sił na interfejsie (`interface_force_balance_equation2`), które operują na osobnych naprężeniach (`mechanical_stress2`).
*   **Wielkości materiałowe**: Model wykorzystuje dwa zestawy modułów sprężystości (parametry Lamégo). Podstawowe dla sprzężystości `u1` oraz nowe (`lame_lambda2`, `shear_modulus2`) dla modelu lepkiego `u2`.

## 3. Lista Wprowadzonych Poprawek
W stosunku do oryginalnego pliku udostępnionego przez użytkownika (`new_beginning.py`), wprowadzono **13 krytycznych zmian** naprawiających błędy w kodzie i architekturze dziedziczenia:

### Poprawki API PorePy, Logika i Literówki (1-6, 8, 9)
1.  **Brakujące parametry (Atrybuty SolidConstants)**: Zamiast próbować dodać parametry dynamicznie w głównej pętli (co zwracało klasyczny `AttributeError`), utworzono nową podklasę `ViscoelasticSolidConstants`, która stabilnie rozszerza natywny słownik jednostek o `lame_lambda2` i `shear_modulus2`.
2.  **Literówka we wzorze**: Poprawiono błąd `"lame_lambd2"` w definicji drugiego modułu Younga.
3.  **Przestarzałe API `initialize_data`**: Naprawiono składnię rejestrowania dyskretyzacji (`update_discretization_parameters`), doprowadzając ją do współczesnego API biblioteki (3 argumenty, zamiast 4). 
4.  **Błędny tensor sztywności**: W funkcji dyskretyzującej naprężenia zmodyfikowano użycie domyślnego `stiffness_tensor` na zdefiniowany nowy `stiffness_tensor2`.
5.  **Kolizja nazw konfiguracyjnych**: Zmieniono nazwę `SolutionStrategyMomentumBalance` na `SolutionStrategyU2`, ponieważ skrypt bazowy PorePy zawiera w sobie identycznie zdefiniowaną klasę bazową, co doprowadzało do nadpisania.
6.  **Unikalny klucz naprężeń (`stress2_keyword`)**: Rozwiązano ukryty konflikt danych – "lepki" człon naprężeniowy bazował na nazwie `"mechanics"`, dokładnie tak samo jak człon sprężysty. Słowniki danych PorePy wzajemnie nadpisywały się w pamięci. Zmieniono identyfikator u2 na `"mechanics2"`.
8.  **Brakujące Type-Hinty**: Poprawiono deklarację typów parsera AD w celu ustabilizowania wglądów operatorów na klasy.
9.  **Brakujące BC/IC**: Uzupełniono strukturę o rejestrację warunków początkowych. Dzięki temu zmienna `u2` poprawnie ma przypisywane zero w `iterate_index=0`.

### Problemy Architektury MRO i Budowania Macierzy (7, 10-13)
Najtrudniejszym problemem z oryginalnego skryptu było nieuruchamianie się solvera we wszystkich fazach na skutek błędnego działania algorytmu **C3 Linearization (MRO)** w Pythonie, odpowiadającego za wielodziedziczenie. Odkryto i załatano problemy skutkujące ominięciem dodawanych równań `u2` podczas zestawiana macierzy z automatycznym różniczkowaniem:

7.  **Złe dziedziczenie Mixinów (Pure Pattern)**: Skrypt rozszerzający mylnie dziedziczył poszczególne pakiety dodatkowe z `pp.PorePyModel`, podczas gdy ostateczny konstruktor i tak już je importuje, psuło to kompatybilność hierarchiczną drzewa.
10. **Problem pętli wywołań w MRO `super()`**: Pakiety `EquationsU2`, `VariablesU2`, `InitialConditionsU2` wyczyszczono z dziedziczenia podstawowych wzorców (jak `pp.VariableMixin`). Aby autorskie nowe obiekty zdołały się poprawnie zarejestrować, Python musi zaaplikować poszczególne nowe obiekty przed ich systematycznymi bazami, więc te nowe części muszą zachowywać postać płaskich modułów ("wisiać w próżni" klasowej).
11. **Spłaszczone Dodawanie Ostateczne**: Zrezygnowano z zamykania funkcjonalności modułu `u2` wewnątrz `ViscoelasticMixin`. By wstrzyknąć 5 nowo napisanych części na początek listy ewaluacji (`set_equations`, `create_variables`), klasy musiały otrzymać swoje natywne deklaracje prosto we wnętrzu końcowej, najważniejszej klasy `ViscoelasticMomentumBalance`. Skutkuje to zmuszeniem interpretera, do wykonania nadpisu zmiennych lepkościowych przed zaaplikowaniem terminalnych definicji z bazowego PorePy, zatrzymujących pętlę.
13. **Zbalansowanie Wektora Wymuszeń (KeyError u2 w Gridzie Krawędziowym)**: Po uzdrowieniu przepływu, wykryto, iż system równań jest asymetryczny pod kątem krawędzi układu (BoundaryGrid). PorePy natywnie nadpisuje parametry krańcowe u1 dyskretyzatorem, jednak `u2` z uwagi na miano "nowego" w solverze – traciło wewnątrz parsera orientację co do wartości granic. Metodą w `update_boundary_values_primary_variables` zarejestrowano w warunkach brzegowych pętlę dla klucza `displacement2_variable`. Wynikiem tego otrzymano wybalansowaną prostokątną i symetryczną ostatecznie zbudowaną macierz (Rozmiar `(648, 648)` – stanowiąc `324` lokalizacje dla standardowego wymiaru `u` dodane do precyzyjnych `324` wymiarów parametru `u2`).

## 4. Struktura logiczna nowego pliku `miki_fix.py`
Zgodnie z w.w łatkami przebudowano logicznie skrypt pod łatwe modyfikacje i czytelność. Składa się obecnie z dedykowanych 13 segmentów blokowych oddzielonych w kodzie od siebie w następującej strukturze:

1. `ViscoelasticSolidConstants` - Definicje stałych materiałowych
2. `GeometryMixin` - Zarządca meshów (Siatka 2D z wymiarem boków, symetryczna)
3. `ViscousElasticModuli` - Obliczenia Lamégo, Tensor Sztywności na bazie wejściowej lepkości
4. `MechanicalStressU2` - Serce operacji (Odpowiednik systemowego Linear Elastic dla `u2`), implementujący zbiór operatorów MPSA
5. `ConstitutiveLawsU2` - Wiązanie fizyk modułu drugiego i delegacja grawitacji
6. `VariablesU2` - Budowa i alokacja zasobów dla zmiennej przemieszczenia oraz interfejsu lepkości między węzłami
7. `EquationsU2` - Centralny rejestrator różniczek i zestawianie całek objętościowych/kontaktowych we wczesnej warstwie czasowej
8. `BoundaryConditionsMixin` - Wyznaczenie na siatkach punktowo typów Neumann/Dirichlet z przypisaniem ciśnień granicznych 5 Pa 
9. `InitialConditionsU2` - Bazowa synchronizacja modelu układów 
10. `SolutionStrategyU2` - Menadżer strategii różniczkowania
11. `BodyForceMixin` - Konkretne wstrzyknięcie maskujące oddziałującej siły ujemnego przyspieszenia we wnętrzu 0.3x0.7 sześcianu sił
12. (usunięte dla poprawy MRO)
13. `ViscoelasticMomentumBalance` - Centrala Ostateczna (Silnik Skryptu) 

## 5. Jak Uruchomić i Sprawdzić?
1. Aktywuj swoje środowsiko Pythona i uruchom przebudowany plik po prostu jako główny skrypt. 
   ```bash
   python miki_fix.py
   ```
2. Skrypt automatycznie podstawi obiekt wbudowaną funkcją i wyświetli stos alertów logowania o prawidłowej iteracji solvera `pypardiso / scipy`. 
3. W module graficznym zostanie utworzone pole siatkowe `displacement_u2.png`, które dodatkowo ukaże się natywnie oknem `matplotlib` (O ile nie użyjesz do tego pracy w tle środowiska konsolowego). Wyświetlona wizualizacja dokumentuje udane obliczenia na macierzach i ukazuje niezerowe wartości `u2` względem warstwy objętości. U1 również jest obliczane poprawnie w tle według natywnych wartości w `MomentumBalance`. 
