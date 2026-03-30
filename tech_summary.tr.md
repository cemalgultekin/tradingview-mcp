## TradingView MCP Sunucusu — Tam Yetenek Analizi

### Temel Mimari
**4.000+ satir MCP sunucusu** ([server.py](src/tradingview_mcp/server.py)), stdio veya HTTP aktarim protokolleri uzerinden **46 arac** sunan, tamamen **bagimsiz saf-Python gosterge/geriye donuk test motoru** (pandas, numpy veya scikit-learn kullanmaz) ve **19 ozellestirilmis detektor modulu** ile desteklenen bir sistemdir.

---

### 11 Kategoride 46 MCP Araci

**Tarama ve Filtreleme (6 arac)**
- `top_gainers` / `top_losers` — Bollinger gostergeleri ile yuzde degisime gore siralama
- `bollinger_scan` — kirilma firsatlari icin dusuk BBW sikisma tespiti
- `rating_filter` — guclu al/sat derecelendirmesine gore filtreleme (-3 ile +3 arasi)
- `volume_breakout_scanner` — hacim sivrilmesi + fiyat kirilmasi kombinasyonlari
- `smart_volume_scanner` — hacim + RSI + teknik gostergeler birlesik tarama

**Derinlemesine Analiz (5 arac)**
- `coin_analysis` — tam teknik analiz (RSI, MACD, SMA, EMA, ATR, Bollinger, OBV, Stochastic, ADX, destek/direnc seviyeleri, piyasa yapisi)
- `consecutive_candles_scan` / `advanced_candle_pattern` — momentum onayiyla mum formasyonu tespiti
- `volume_confirmation_analysis` — hacim-fiyat uyumsuzluk sinyalleri
- `multi_agent_analysis` — **3-ajanli tartisma sistemi** (Teknik, Duygu Durumu ve Risk ajanlari tartisarak uzlasiya varir)

**Misir Borsasi Ozel Araclar (7 arac)**
- Tam EGX ekosistemi: piyasa genel gorunumu, sektor taramasi (piyasa degeri agirliklari ile 18 sektor), endeks analizi, hisse tarayici (buyume/deger/momentum/kalite), islem plani olusturucu, Fibonacci geri cekilmesi — tumune [egx_sectors.py](src/tradingview_mcp/core/data/egx_sectors.py) icerisindeki sabit sektor meta verileriyle desteklenir

**Coklu Zaman Dilimi ve Duygu Analizi (4 arac)**
- `multi_timeframe_analysis` — tek seferde 5d/15d/1s/4s/1G karsilastirmasi
- `market_sentiment` — proxy rotasyonu ile **Reddit kazima** (herkese acik JSON uzerinden)
- `financial_news` — CoinDesk, CoinTelegraph, Reuters RSS akisi toplama
- `combined_analysis` — TradingView + Reddit + Haber birlesik **konfluans araci**

**Geriye Donuk Test ve Dogrulama (5 arac)**
- `backtest_strategy` — islem kayitlari, sermaye egrisi, komisyon/kayma modellemesi ile tam geriye donuk test
- `compare_strategies` — 6 stratejinin tamamini al-ve-tut karsilastirmaliyla siralayan liderlik tablosu
- `walk_forward_backtest_strategy` — egitim/test bolumleri ve saglam puan ile **asiri uyum tespiti**
- `batch_walk_forward_test` — tek seferde 50 sembole kadar **capraz para birimi ileri yurume** dogrulamasi
- `out_of_sample_test` — temiz kronolojik egitim/test bolunmesi ile **saf orneklem disi test**

**Guvenlik ve Sinyal Butunlugu (2 arac)**
- `rug_pull_detector` — cokus olaylari, balina bosaitmalarini, pompala-ve-bosat kaliplarini, likidite sagligini tespit eder; risk puani (0-100) ve ciddiyet derecesi (KRITIK/YUKSEK/ORTA/DUSUK) dondurur
- `repaint_detector` — mum mum artimli sinyaller ile tam bakis acisiyla hesaplanan sinyalleri karsilastirarak sinyal kararliligini dogrular; hayalet, kaybolan ve ters donen sinyalleri tespit eder

**Piyasa Zekasi Dedektorleri (12 arac)**
- `divergence_detector` — 3 gosterge (RSI, MACD histogrami, OBV) uzerinde 4 uyumsuzluk turu tespit eder (yukselis uyumsuzlugu, dusus uyumsuzlugu, gizli yukselis uyumsuzlugu, gizli dusus uyumsuzlugu); sallanma noktasi eslestirme ve guc puanlamasi ile
- `wash_trade_detector` — 5 metrik ile sahte hacim tespiti: hacim-fiyat Pearson korelasyonu, hacim kumelenmesi (tekrarlanan ayni mumlar), ardisik benzeri oran, yuvarlak sayili hacim orani, log-hacim dagilim tekduzesi
- `correlation_detector` — yapilandirabilir karsilastirma olcutlerine karsi Pearson korelasyonu, beta (piyasa duyarliligi), yilliklandirilmis alfa (fazla getiri); baglantiyi kesme olayi tespiti ile yuvarlanan korelasyon; bagimsizlik puani
- `volatility_regime_detector` — ATR yuzdelik dilimi, Bollinger Bant Genisligi yuzdelik dilimi, gunluk aralik yuzdelik dilimi, getiri oynaklik yuzdelik dilimi bileseni ile DUSUK/NORMAL/YUKSEK/ASIRI oynaklik siniflandirma; rejim basina strateji onerileri; Bollinger sikismalari ve rejim gecisleri tespiti
- `stop_hunt_detector` — sallanma yuksek/alcak pivot seviyelerini surukliyip sonra ters donen mumlarda fitil-govde orani analizi ile likidite tuzagi tespiti; hacim dogrulamasi; %2 fiyat bantlari icinde av bolgesi kumelenmesi; siklik puanlamasi
- `dead_cat_bounce_detector` — cokus olaylarini (yapilandirabilir esik) tanimlar, sonraki sicramalari tespit eder, her birini su kriterlere gore olgun kedi veya gercek toparlanma olarak siniflandirir: Fibonacci geri cekilme derinligi, sicrama sirasinda hacim trendi, sicrama tepesinde RSI, sicrama sonrasi fiyat hareketi
- `accumulation_distribution_detector` — OBV egimi ve Chaikin A/D cizgisi egiminin fiyat egimiyle yuvarlanan pencereler uzerinden karsilastirilmasi yoluyla Wyckoff tarzi faz tespiti (BIRIKIM, DAGITIM, YUKSELIS, DUSUS, BIRIKIM_UYUMSUZLUGU, DAGITIM_UYUMSUZLUGU, NOTR)
- `slippage_risk_detector` — yapilandirabilir pozisyon buyuklukleri ($1K/$5K/$10K/$50K/$100K) icin piyasa etki modeli (spread/2 + k x karekkok(katilim_orani)) kullanan gercekci kayma tahmini; likidite katmani siniflandirmasi (COK YUKSEK/YUKSEK/ORTA/DUSUK/COK DUSUK); geriye donuk test varsayimi karsilastirmasi
- `market_regime_classifier` — ADX (trend gucu), +DI/-DI (yon), Kaufman Verimlilik Orani (gurultu), EMA20/EMA50 hizalanmasi, dalgalanma endeksi ile YUKSELIS_TRENDI, DUSUS_TRENDI, YATAY veya DALGALI siniflandirma; strateji basina uygunluk puanlari (0.0–1.0)
- `arbitrage_detector` — araclar arasi normalize spread analizi; z-puanlari hesaplar, spread 1.5 standart sapmayi astiginda aktif firsatlari tespit eder; gecmis arbitraj pencerelerini izler; karsilastirma ciftlerini otomatik secer (BTC-USD - GBTC/BITO/IBIT, ETH-USD - ETHE/ETHA)
- `news_price_lag_detector` — duygu-fiyat hizalama kontrolu (HIZALI/UYUMSUZ/NOTR), hacim sivrilmesi devam analizi (devam - geri donus orani), haber islem yapilabilirligi degerlendirmesi (YUKSEK/ORTA/DUSUK), coklu ufuk momentum (1g/3g/7g/14g/30g)
- `seasonality_detector` — haftanin gunu analizi (Pazartesi'den Pazar'a ortalama getiri, kazanma orani, genel uzerindeki avantaj), yilin ayi analizi (Ocak'tan Aralik'a), ayin ilk yarisi ile ikinci yarisi karsilastirmasi, her kalip icin bootstrap istatistiksel anlamlilik testi (p-degerleri)

**Formasyon Tanima (3 arac)**
- `candlestick_pattern_scanner` — uc kategoride 25'ten fazla mum formasyonu tespit eder:
  - *Tek mum (9):* Doji, Yusufcuk Doji, Mezar Tasi Doji, Cekic (Hammer), Ters Cekic (Inverted Hammer), Kayan Yildiz (Shooting Star), Yukselis Marubozu, Dusus Marubozu, Topaclama (Spinning Top)
  - *Cift mum (10):* Yukselis Yutma (Bullish Engulfing), Dusus Yutma (Bearish Engulfing), Delici Cizgi (Piercing Line), Kara Bulut Ortme (Dark Cloud Cover), Cimbiz Tepesi (Tweezer Top), Cimbiz Dibi (Tweezer Bottom), Yukselis Harami, Dusus Harami, Yukselis Harami Capraz, Dusus Harami Capraz
  - *Uc mum (6):* Sabah Yildizi (Morning Star), Aksam Yildizi (Evening Star), Uc Beyaz Asker (Three White Soldiers), Uc Kara Karga (Three Black Crows), Uc Icsel Yukselis (Three Inside Up), Uc Icsel Dusus (Three Inside Down)
  - Her formasyon guvenilirlik derecesi (dusuk/orta/yuksek), RSI baglamini ve yukselis/dusus/notr siniflandirmasini icerir
- `chart_formation_scanner` — 17 klasik coklu-mum yapisal formasyon tespit eder:
  - *Geri Donus (8):* Omuz Bas Omuz (Head and Shoulders), Ters Omuz Bas Omuz (Inverse Head and Shoulders), Ikili Tepe (Double Top), Ikili Dip (Double Bottom), Uclu Tepe (Triple Top), Uclu Dip (Triple Bottom), Yukselen Kama (Rising Wedge), Dusen Kama (Falling Wedge)
  - *Devam (6):* Yukselen Ucgen (Ascending Triangle), Alcalan Ucgen (Descending Triangle), Boga Bayragi (Bull Flag), Ayi Bayragi (Bear Flag), Flama (Pennant), Fincan ve Kulp (Cup and Handle)
  - *Kirilma (1):* Simetrik Ucgen (Symmetrical Triangle)
  - *Trend Yapisi (3):* Yukselen Kanal (Ascending Channel), Alcalan Kanal (Descending Channel), Yatay Kanal (Horizontal Channel)
  - Her formasyon fiyat hedeflerini (olculmus hareket), boyun cizgilerini, kirilma seviyelerini, R² trend cizgisi uyum kalitesini ve kanallar icindeki mevcut konumu icerir
- `support_resistance_mapper` — coklu temas pivot kumelenmesi ile D/D bolgelerini haritalandirir; guc puanlamasi (0-10), en yakin seviye tespiti, konum degerlendirmesi (destege yakin / dirence yakin / orta bolge), hacim dogrulamali kirilma/kirilim dogrulamasi ve zarar durdurma boyutlandirmasi icin ATR baglami

**Piyasa Verileri (2 arac)**
- `yahoo_price` — hisse senetleri, kripto, ETF'ler, doviz, endeksler icin anlik OHLCV
- `market_snapshot` — kuresel genel gorunum (buyuk endeksler, en iyi kripto, doviz kurlari, onemli ETF'ler)

---

### 6 Yerlesik Islem Stratejisi
| Strateji | Mantik |
|----------|--------|
| RSI | Asiri satim <30'da al, asiri alim >70'de sat |
| Bollinger | Alt banttan al, orta banttan sat |
| MACD | Altin caprazda al (golden cross), olum caprazinda sat (death cross) |
| EMA Capraz | EMA20/EMA50 caprazlama |
| Supertrend | ATR tabanli trend donusu |
| Donchian | N-periyot yuksek/alcak kirilmasi |

Her biri su metrikleri uretir: Sharpe orani, Calmar orani, maksimum dusus (drawdown), kar faktoru, kazanma orani, beklenti, en iyi/en kotu islem, tam islem kaydi ve sermaye egrisi verileri.

---

### Rug Pull (Hali Cekme) Tespiti
`rug_pull_detector` ([security_checks.py](src/tradingview_mcp/core/services/security_checks.py)) sunlari analiz eder:
- **Cokus olaylari** — hacim sivrilerisi ile esik degeri asan tek-mum dususleri
- **Balina aktivitesi** — yon siniflandirmasi ile anormal hacim (>4.5x SMA20) (pompala/bosat/notr)
- **Pompala-ve-bosat kaliplari** — bir pencere icinde hizli yuksselisler (>%50) ardindan sert dususler
- **Likidite sagligi** — hacim tutarliligi, trend, ortalama spread, sifir hacimli gun orani
- 0-100 arasi bilesk risk puani ile KRITIK/YUKSEK/ORTA/DUSUK ciddiyet derecesi uretir

### Repaint (Yeniden Boyama) Tespiti
`repaint_detector` ([repaint_detector.py](src/tradingview_mcp/core/services/repaint_detector.py)) sinyal butunlugunu su sekilde dogrular:
- Her stratejiyi mum mum calistirarak (yalnizca N. muma kadar olan verileri gorerek) canli islem simule eder
- Bu "canli" sinyalleri tam bakis acisiyla hesaplanan sinyallerle karsilastirir
- Uyumsuzluklari **hayalet** (yalnizca bakis acisinda), **kaybolan** (bakis acisinda kaybolan) veya **ters donen** (yon degistiren) olarak siniflandirir
- 0.0–1.0 arasi kararlili puani ve risk seviyesi (TEMIZ/BUYUK OLCUDE TEMIZ/ORTA/ONEMLI/SIDDETLI) dondurur

### Formasyon Tanima

#### Mum Formasyonlari ([candlestick_patterns.py](src/tradingview_mcp/core/services/candlestick_patterns.py))
OHLCV verilerinden 25'ten fazla formasyonun saf-Python ile tanima, her biri tur/sinyal/guvenilirlik meta verileri ile:

| Kategori | Formasyonlar |
|----------|-------------|
| **Tek mum geri donus** | Cekic (Hammer), Ters Cekic (Inverted Hammer), Kayan Yildiz (Shooting Star) |
| **Tek mum momentum** | Yukselis Marubozu (Bullish Marubozu), Dusus Marubozu (Bearish Marubozu) |
| **Tek mum kararsizlik** | Doji, Yusufcuk Doji (Dragonfly Doji), Mezar Tasi Doji (Gravestone Doji), Topaclama (Spinning Top) |
| **Cift mum geri donus** | Yukselis Yutma (Bullish Engulfing), Dusus Yutma (Bearish Engulfing), Delici Cizgi (Piercing Line), Kara Bulut Ortme (Dark Cloud Cover), Cimbiz Tepesi (Tweezer Top), Cimbiz Dibi (Tweezer Bottom), Yukselis Harami (Bullish Harami), Dusus Harami (Bearish Harami), Yukselis Harami Capraz (Bullish Harami Cross), Dusus Harami Capraz (Bearish Harami Cross) |
| **Uc mum geri donus** | Sabah Yildizi (Morning Star), Aksam Yildizi (Evening Star), Uc Icsel Yukselis (Three Inside Up), Uc Icsel Dusus (Three Inside Down) |
| **Uc mum momentum** | Uc Beyaz Asker (Three White Soldiers), Uc Kara Karga (Three Black Crows) |

Her formasyon sunlari icerir: mum endeksi, tarih, kapanis fiyati, RSI baglami, guvenilirlik derecesi (dusuk/orta/yuksek) ve yukselis/dusus/notr siniflandirmasi. Sonuclar yakin gecmis yonelimini ve formasyon siklik dagilimini ozetler.

#### Grafik Formasyonlari ([chart_formations.py](src/tradingview_mcp/core/services/chart_formations.py))
Sallanma yuksek/alcak tespiti ve geometrik uydurma ile yapisal formasyon tanima:

| Formasyon | Tur | Sinyal | Tespit Yontemi |
|-----------|-----|--------|----------------|
| **Omuz Bas Omuz (Head & Shoulders)** | Dusussel | Geri Donus | Ortadaki en yuksek olan 3 sallanma tepesi, kabaca esit omuzlar, ara dipler arasinda boyun cizgisi |
| **Ters Omuz Bas Omuz (Inverse H&S)** | Yukselissel | Geri Donus | Sallanma dipleri uzerinde OBO'nun aynasi |
| **Ikili Tepe (Double Top)** | Dusussel | Geri Donus | Araya giren dip ile ayni seviyede 2 sallanma tepesi |
| **Ikili Dip (Double Bottom)** | Yukselissel | Geri Donus | Araya giren tepe ile ayni seviyede 2 sallanma dibi |
| **Uclu Tepe (Triple Top)** | Dusussel | Geri Donus | Araya giren diplerle ayni seviyede 3 sallanma tepesi |
| **Uclu Dip (Triple Bottom)** | Yukselissel | Geri Donus | Araya giren tepelerle ayni seviyede 3 sallanma dibi |
| **Yukselen Ucgen (Ascending Triangle)** | Yukselissel | Devam | Yatay direnc + yukselen destek trend cizgisi (dogrusal regresyon) |
| **Alcalan Ucgen (Descending Triangle)** | Dusussel | Devam | Yatay destek + dusen direnc trend cizgisi |
| **Simetrik Ucgen (Symmetrical Triangle)** | Notr | Kirilma | Yakinlasan trend cizgileri (her ikisi daralan) |
| **Yukselen Kama (Rising Wedge)** | Dusussel | Geri Donus | Her iki trend cizgisi yukselen ama yakinlasan |
| **Dusen Kama (Falling Wedge)** | Yukselissel | Geri Donus | Her iki trend cizgisi dusen ama yakinlasan |
| **Boga Bayragi (Bull Flag)** | Yukselissel | Devam | Guclu yukari itki + kucuk asagi egimli konsolidasyon |
| **Ayi Bayragi (Bear Flag)** | Dusussel | Devam | Guclu asagi itki + kucuk yukari egimli konsolidasyon |
| **Flama (Pennant)** | Her ikisi | Devam | Itki + yakinlasan konsolidasyon |
| **Fincan ve Kulp (Cup & Handle)** | Yukselissel | Devam | Benzer seviyede kenarlara sahip U sekilli taban + kucuk kulp geri cekilmesi |
| **Yukselen Kanal (Ascending Channel)** | Yukselissel | Trend | Mevcut konum takibi ile paralel yukselen trend cizgileri |
| **Alcalan Kanal (Descending Channel)** | Dusussel | Trend | Paralel dusen trend cizgileri |
| **Yatay Kanal (Horizontal Channel)** | Notr | Aralik | Paralel yatay trend cizgileri |

Her formasyon sunlari icerir: fiyat hedefleri (olculmus hareket), boyun cizgileri/kirilma seviyeleri, trend cizgileri icin R² uyum kalitesi ve kanallar icindeki mevcut konum.

#### Destek / Direnc Haritalama ([support_resistance.py](src/tradingview_mcp/core/services/support_resistance.py))
Puanlama ile coklu temas bolgesi tespiti:
- **Pivot tespiti** — yapilandirabilir geriye bakis hassasiyeti ile sallanma yuksekleri/alcaklari
- **Bolge kumelenmesi** — yakin pivotlari tolerans yuzdesine gore bolgelere gruplar (tek fiyat noktalari degil)
- **Guc puanlamasi (0-10)** — su kriterlere dayanir: temas sayisi (fazla = guclu), temas noktalarindaki hacim / ortalama hacim, temaslarin yakinligi
- **Konum degerlendirmesi** — en yakin destek/direnc, her birine uzaklik, aralik icindeki konum (destege yakin/dirence yakin/orta bolge)
- **Kirilma dogrulamasi** — hacim dogrulamasi ile D/D bolgelerinin yakin kirilma/kirilimlarini tespit eder
- **ATR baglami** — seviyelere gore zarar durdurma boyutlandirmasi icin mevcut oynaklik

---

### Toplu Ileri Yurume ve Orneklem Disi Test
- `batch_walk_forward_test` — bir stratejiyi ayni anda birden fazla para birimi/sembol uzerinde test eder, varyans analiziyle capraz sembol saglamligini toplar
- `out_of_sample_test` — bozulma orani ve Sharpe bozulma puanlamasi ile temiz kronolojik bolunme (ornegin %70 egitim / %30 en son verilerle test)

### Piyasa Zekasi Dedektorleri

| Detektor | Ne Tespit Eder | Temel Cikti |
|----------|----------------|-------------|
| [divergence_detector.py](src/tradingview_mcp/core/services/divergence_detector.py) | RSI/MACD/OBV fiyat uyumsuzluklari | Guc puanlariyla yukselis/dusus/gizli uyumsuzluklar |
| [wash_trade_detector.py](src/tradingview_mcp/core/services/wash_trade_detector.py) | Sahte/yapay hacim | Yikama islemi olasiligi (0-100), kanit listesi |
| [correlation_detector.py](src/tradingview_mcp/core/services/correlation_detector.py) | Karsilastirma olcutu korelasyonu/beta/alfa | Bagimsizlik puani, ayrisma olaylari |
| [volatility_regime.py](src/tradingview_mcp/core/services/volatility_regime.py) | Oynaklik rejimi siniflandirmasi | DUSUK/NORMAL/YUKSEK/ASIRI + strateji onerileri |
| [stop_hunt_detector.py](src/tradingview_mcp/core/services/stop_hunt_detector.py) | Likidite tuzaklari / stop avlari | Av kumeleri, suruklenen D/D seviyeleri, siklik puani |
| [dead_cat_detector.py](src/tradingview_mcp/core/services/dead_cat_detector.py) | Cokusler sonrasi basarisiz rahatlama sicrmalari | Olgun kedi - gercek toparlanma siniflandirmasi |
| [accumulation_detector.py](src/tradingview_mcp/core/services/accumulation_detector.py) | Akilli para birikimi/dagitimi | Wyckoff tarzi faz tespiti, B/D bolgeleri |
| [slippage_risk.py](src/tradingview_mcp/core/services/slippage_risk.py) | Pozisyon buyuklugu basina gercekci kayma | Likidite katmani, onerilen geriye donuk test ayarlari |
| [regime_classifier.py](src/tradingview_mcp/core/services/regime_classifier.py) | Piyasa rejimi (trendli/yatay/dalgali) | ADX/ER tabanli siniflandirma, strateji uygunluk puanlari |
| [arbitrage_detector.py](src/tradingview_mcp/core/services/arbitrage_detector.py) | Araclar arasi fiyat farkliliklari | Spread z-puanlari, arbitraj pencereleri |
| [news_lag_detector.py](src/tradingview_mcp/core/services/news_lag_detector.py) | Haber-fiyat iliskisi | Duygu hizalanmasi, islem yapilabilirlik degerlendirmesi |
| [seasonality_detector.py](src/tradingview_mcp/core/services/seasonality_detector.py) | Gun/ay mevsimsel kaliplari | Bootstrap-testli anlamli kalipler |
| [candlestick_patterns.py](src/tradingview_mcp/core/services/candlestick_patterns.py) | 25+ mum formasyonu | Formasyon turu, sinyal, guvenilirlik, RSI baglami |
| [chart_formations.py](src/tradingview_mcp/core/services/chart_formations.py) | 17 klasik grafik formasyonu | Hedefler, boyun cizgileri, kirilma seviyeleri, R² uyumu |
| [support_resistance.py](src/tradingview_mcp/core/services/support_resistance.py) | Coklu temas D/D bolgeleri | Guc puanlari, konum degerlendirmesi, kirilmalar |

---

### Belirgin Olmayan Yetenekler

1. **Saf-Python gosterge kutuphanesi** ([indicators_calc.py](src/tradingview_mcp/core/services/indicators_calc.py)) — EMA, SMA, RSI (Wilder duzenlemesi), Bollinger, MACD, ATR, Supertrend, Donchian, ADX — hepsi sifirdan, sifir bagimlilik ile
2. **Proxy rotasyon sistemi** ([proxy_manager.py](src/tradingview_mcp/core/services/proxy_manager.py)) — Reddit/Yahoo hiz siniri bypass'i icin oturum havuzlu (1-250 oturum) Webshare konut proxy entegrasyonu
3. **Otomatik piyasa turu tespiti** — hisse senedi ile kripto sembollerini otomatik olarak ayirt eder ve dogru tarayiciya yonlendirir
4. **Toplu isleme** — akilli geri donuslerle API cagrisi basina 200-500 sembol isler
5. **22 borsa sembol listesi** [coinlist/](src/tradingview_mcp/coinlist/) icinde, Misir, Turkiye, ABD, Malezya, Hong Kong genelinde 11 kripto + 11 hisse senedi borsasi
6. **OpenClaw CLI sarmalayici** ([openclaw/trading.py](openclaw/trading.py)) — MCP baglami disinda araclari kullanmak icin bagimsiz Python komut satiri arayuzu
7. **Ileri yurume saglamlik puanlamasi** — orneklem disi / orneklem ici performans oranina gore stratejileri SAGLAM/ORTA/ZAYIF/ASIRI UYUMLU olarak isaretler
8. **Coklu ajan tartismasi** — `multi_agent_analysis` araci 3 yapay zeka kisiligini (Teknik Analist, Duygu Analisti, Risk Yoneticisi) bagimsiz degerlendirip uzlasi sentezleyen bir tartismaya sokar
9. **Cift aktarim** — stdio (Claude Code icin) veya streamable-http (Docker/uzaktan icin) olarak saglik kontrolu ucu noktasi ile calisir
10. **Sifir API anahtari gereksinimi** — her sey herkese acik API'ler ile kutudan cikar calisir; proxy kimlik bilgileri yogun kullanim icin istege baglidir

---

### Piyasa Kapsami
**Kripto:** Binance, Bybit, Bitget, OKX, Coinbase, KuCoin, Gate.io, Huobi, Bitfinex
**Hisse Senetleri:** NASDAQ, NYSE, EGX (Misir), BIST (Turkiye), BURSA/KLSE/MYX/ACE/LEAP (Malezya), HKEX (Hong Kong)
**Diger:** ETF'ler, doviz ciftleri, buyuk endeksler (S&P 500, Dow, NASDAQ Composite)
