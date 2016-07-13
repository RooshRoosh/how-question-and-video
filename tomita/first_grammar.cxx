#encoding "utf8"

ShortTermRus -> Word<wfl="[А-Яа-я-]{1,9}">(Word<wfl="[0-9]{1,4}">);
ShortTermEng -> Word<wfl="[A-Za-z0-9-]*">(Word<wfl="[0-9]{1,4}">);
ShortTerm-> ShortTermEng|ShortTermRus;
//PersonalPronoun -> "себе" | "себя" | "тебя" | "тебе";
PrepT -> "c"|Prep;
NounGroup -> (PrepT) (Adj) ShortTerm|Noun;
VerbLike -> Word<gram="V"> | Word<wfl=".+(ться|чься)">{outgram='V,inf,ipf'} | Word<wfl=".+(чь|ть)">{outgram='V,inf,pf'};
SimpleQuestionStart -> 'как' (Word<gram="~V">+ interp(Q.FirstWords::not_norm)) VerbLike interp(Q.Verb::not_norm) (NounGroup interp(Q.NounAfterVerb::not_norm));
