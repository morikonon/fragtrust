| Split | Blocks | Per class | Used for |
|---|---|---|---|
| train | 1,500,000 | 20,000–20,000 | fitting |
| val_a | 150,000 | 2,000–2,000 | early stopping, grid, temperature |
| val_b | 150,000 | 2,000–2,000 | thresholds, conformal quantiles |
| test | 768,000 | 9,995–10,459 | all reported numbers |
| ood_aes_ctr | 20,000 | – | AES-256-CTR encryption of randomly chosen test blocks; one key, fresh random nonce per block |
| ood_random | 20,000 | – | uniformly random bytes |
