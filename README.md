# Case-Declining


# Learning curves
## ngrams
- hidden_units = 60
- activation = 'softmax'
- epochs = 350

| Model type      | Value       | Validation Score   | Generalization Score |
|-----------------|-------------|--------------------|----------------------|
| Encoder-Decoder | (4, 15, 15) | 0.5503212081491045 | 0.6380378214659013   |
| Encoder-Decoder | (4, 3, 3)   | 0.7937005259840998 | 0.7761349821942418   |
| Encoder-Decoder | (3, 15, 15) | 0.5503212081491045 | 0.7051654294812815   |
| Encoder-Decoder | (3, 3, 3)   | 0.7937005259840998 | 0.7614874157334882   |
| Encoder-Decoder |**(3, 2, 2)**|**0.7937005259840998**|**0.8105211970864157**|
| Vanilla         | (4, 15, 15) | 0.5503212081491045 | 0.5923018455077502   |
| Vanilla         | (4, 3, 3)   | 0.7937005259840998 | 0.7539474411291538   |
| Vanilla         | (3, 15, 15) | 0.6299605249474366 | 0.6484993172480279   |
| Vanilla         | (3, 3, 3)   | 0.7937005259840998 | 0.7688809598286291   |
| Vanilla         |**(3, 2, 2)**|**0.7937005259840998**|**0.8038772818939559**|

## dropout
- activation = 'softmax'
- epochs = 650
- hidden_units = 80
- ngram_factors = (3, 2, 2)

| Model type      | Value       | Validation Score   | Generalization Score |
|-----------------|-------------|--------------------|----------------------|
| Encoder-Decoder | 0.          | 0.7937005259840998 | 0.738397678969255   |
| Encoder-Decoder | 0.05        | 0.7937005259840998 | 0.7902496105469656   |
| Encoder-Decoder | 0.1         | 0.7937005259840998 | 0.7832558843809432   |
| Encoder-Decoder | 0.15        | 0.7937005259840998 | 0.7971216912249737   |
| Encoder-Decoder | 0.2         | 0.7937005259840998 | 0.8038772818939559   |
| Encoder-Decoder | 0.25        | 0.7937005259840998 | 0.8105211970864157   |
| Encoder-Decoder | 0.3         | 0.7937005259840998 | 0.7971216912249737   |
| Encoder-Decoder | 0.35        | 0.7937005259840998 | 0.8105211970864157   |
| Encoder-Decoder |**0.4**      | 0.7937005259840998 |**0.8234917326321642**|
| Vanilla         | 0.          | 0.7937005259840998 | 0.7902496105469656   |
| Vanilla         | 0.05        | 0.7937005259840998 | 0.8170579406432438   |
| Vanilla         | 0.1         | 0.7937005259840998 | 0.8360660648932765   |
| Vanilla         | 0.15        | 0.7937005259840998 | 0.7971216912249737   |
| Vanilla         |**0.2**        | 0.7937005259840998 |**0.842213830272317**|
| Vanilla         | 0.25        | 0.7937005259840998 | 0.8105211970864157   |
| Vanilla         | 0.3         | 0.7937005259840998 | 0.8234917326321642   |
| Vanilla         | 0.35        | 0.7937005259840998 | 0.7971216912249737   |
| Vanilla         | 0.4         | 0.7937005259840998 | 0.8038772818939559   |

## hidden ubits
- activation = 'softmax'
- epochs = 650
- dropout = 0.4
- ngram_factors = (3, 2, 2)

| Model type      | Value       | Validation Score   | Generalization Score |
|-----------------|-------------|--------------------|----------------------|
| Encoder-Decoder | 50          | 0.7937005259840998 | 0.7539474411291538   |
| Encoder-Decoder | 60          | 0.7469007910928608 | 0.7614874157334882   |
| Encoder-Decoder | 70          | 0.7937005259840998 | 0.7832558843809432   |
| Encoder-Decoder | 80          | 0.7937005259840998 | 0.7902496105469656   |
| Encoder-Decoder | 90          | 0.7937005259840998 | 0.7971216912249737   |
| Encoder-Decoder |**100**       | 0.7937005259840998 |**0.8038772818939559**|
| Encoder-Decoder | 110         | 0.7937005259840998 | 0.7462535630467956   |
| Encoder-Decoder | 120         | 0.746900791092860  | 0.7902496105469656   |
| Encoder-Decoder |**130**      |**0.8355496558274308**| 0.7614874157334882   |
| Vanilla         | 50          | 0.7937005259840998 | 0.7832558843809432   |
| Vanilla         | 60          | 0.7937005259840998 | 0.7614874157334882   |
| Vanilla         | 70          | 0.7937005259840998 | 0.7902496105469656   |
| Vanilla         | 80          | 0.7937005259840998 | 0.7688809598286291   |
| Vanilla         |**90**       | 0.7937005259840998 |**0.7971216912249737**|
| Vanilla         | 100         | 0.7937005259840998 | 0.7902496105469656   |
| Vanilla         | 110         | 0.7937005259840998 | 0.7761349821942418   |
| Vanilla         | 120         | 0.7937005259840998 | 0.7902496105469656   |
| Vanilla         | 130         | 0.7937005259840998 | 0.7902496105469656   |
