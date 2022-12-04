# Installation

Install `openjdk`

```
brew install openjdk
sudo ln -sfn $(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

Install `protobuf`

```
brew install protobuf
```

Install python dependencies

```
pip3 install git+https://github.com/neuml/txtai
pip3 install git+https://github.com/neuml/txtai#egg=txtai[pipeline]
```
