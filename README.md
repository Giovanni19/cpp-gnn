# cpp-gnn

A Graph Neural Network implementation from scratch in modern C++ (C++17, no dependencies).

## What's there

- `Graph`, `Node`, `Edge`: directed graph with node and edge features.
- `Linear`, `ReLU`, `MLP`: layers with forward, backward and SGD `step`.
- `MessagePassing`: one GNN layer. An edge MLP turns `[source, target, edge features]`
  into a message, messages are summed (or averaged) per target node, and a node MLP
  turns `[node features, aggregated messages]` into the new node features.
  Supports backward, so layers can be stacked and trained end to end.
- `softmax_cross_entropy` for classification.

## Build and test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Tests use `assert`, so build without `NDEBUG` (the default or `Debug`).

## Examples

| Example | What it does |
|---|---|
| `test_train` | MLP learns XOR |
| `test_train_gnn` | one `MessagePassing` layer learns to sum incoming neighbor features |
| `test_two_layers` | two stacked layers learn a 2-hop target |
| `karate_club` | node classification on Zachary's Karate Club with only 2 labeled nodes |

```bash
./build/karate_club data/karate_club.txt
```

## Next ideas

- Weight decay / Adam
- Faster linear algebra (Eigen)
