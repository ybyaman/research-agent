Kullanıcının araştırma sorusu ve şu ana kadar toplanan bulgular aşağıda.

## Soru
{question}

## Alt sorular ve bulgular
{coverage}

## Görevin
Adım adım değerlendir:

1. assessments: Her alt soru için ayrı bir değerlendirme yaz.
   - answered=true: bulgular o alt soruyu somut olarak cevaplıyor (en az bir rakam, tanım veya net ifade).
     Her ayrıntının olması gerekmez.
   - answered=false: bulgular yok ya da yalnızca "değişir", "bağlıdır" gibi belirsiz ifadeler var.
     Bu durumda missing alanına eksik bilgiyi arayacak kısa bir arama ifadesi yaz.
2. question_answered: Bu bulgularla kullanıcıya asıl sorusu için makul ve gerekçeli bir cevap verilebiliyorsa true.
   Akla gelebilecek her ek konu cevabı engellemez; yalnızca cevap için vazgeçilmez bir bilgi eksikse false ver.
3. extra_gaps: Yalnızca question_answered=false ise doldur. Hiçbir alt soruda olmayan, cevap için vazgeçilmez
   en fazla 2 konu yaz. question_answered=true ise boş liste bırak.
4. notes: Bulgular arasında çelişki veya tek kaynağa dayanan önemli bir iddia varsa bir iki cümleyle yaz; yoksa boş bırak.

Arama ifadeleri soruya özgü olsun ve bulgulardaki konuları tekrar etmesin.
