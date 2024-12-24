from wideresnet_model import WideResNet

def create_wideresnet(num_classes=2):
    # WideResNet-28-10: depth=28, widen_factor=10, dropout_rate=0.3
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)
