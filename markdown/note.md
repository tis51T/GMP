### TWITTER SC

- sample will look alike below before pass to DataLoader
```
[
{
    'sentence': sentence,
    "aesc_spans": [(1, 3, 'POS'), (10, 11, 'NEG')],  # Example spans
    'gt': [('battery life', 'POS'), ('camera', 'NEG')],
    'image_id': image_path,
    'caption': 'Phone review',
    'aspects_num': 2
}, {

}
]
```

- Must have Collator
- aesc_infos is "TWITTER_AE" in batch