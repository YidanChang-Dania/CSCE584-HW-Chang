import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# Use an ASCII-friendly font to avoid rendering issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_figure1():
    """Figure 1: Three-layer ResNet architecture with tensor shapes (layout preserved)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    ax.set_xlim(0, 13)   # widen to prevent right-label clipping
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6.3, 7.5, 'Three-Layer ResNet Architecture and Tensor Shapes',
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))

    # Input block
    input_box = FancyBboxPatch((1, 5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#98e699', edgecolor='black')
    ax.add_patch(input_box)
    ax.text(1.75, 5.5, 'Input\nX(0)\n[B,3,32,32]',
            ha='center', va='center', fontsize=11)

    # Stem conv
    conv0 = FancyBboxPatch((3, 5), 1.2, 1, boxstyle="round,pad=0.1",
                           facecolor='#fff9d6', edgecolor='black')
    ax.add_patch(conv0)
    ax.text(3.6, 5.5, 'Conv\n3->16\nk=3, s=1',
            ha='center', va='center', fontsize=10)
    ax.text(3.6, 6.2, '[B,3,32,32]', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # ResBlock 1
    block1 = FancyBboxPatch((4.8, 4.5), 1.2, 2, boxstyle="round,pad=0.1",
                            facecolor='#f4a3a3', edgecolor='black')
    ax.add_patch(block1)
    ax.text(5.4, 5.5,
            'ResBlock 1\n'
            'Conv->BN->ReLU\n'
            'Conv->BN\n'
            '(add skip)->ReLU\n'
            '16->16\ns=1',
            ha='center', va='center', fontsize=9)
    ax.text(5.4, 6.7, '[B,16,32,32]', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # ResBlock 2
    block2 = FancyBboxPatch((6.6, 3.5), 1.2, 2, boxstyle="round,pad=0.1",
                            facecolor='#f4a3a3', edgecolor='black')
    ax.add_patch(block2)
    ax.text(7.2, 4.5,
            'ResBlock 2\n'
            'Conv->BN->ReLU\n'
            'Conv->BN\n'
            '(add skip)->ReLU\n'
            '16->32\ns=2',
            ha='center', va='center', fontsize=9)
    ax.text(7.2, 5.7, '[B,32,16,16]', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # ResBlock 3
    block3 = FancyBboxPatch((8.4, 2.5), 1.2, 2, boxstyle="round,pad=0.1",
                            facecolor='#f4a3a3', edgecolor='black')
    ax.add_patch(block3)
    ax.text(9.0, 3.5,
            'ResBlock 3\n'
            'Conv->BN->ReLU\n'
            'Conv->BN\n'
            '(add skip)->ReLU\n'
            '32->64\ns=2',
            ha='center', va='center', fontsize=9)
    ax.text(9.0, 4.7, '[B,64,8,8]', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # GAP + FC head
    gap_fc = FancyBboxPatch((9.8, 1.5), 1.2, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#cfe9f7', edgecolor='black')
    ax.add_patch(gap_fc)
    ax.text(10.4, 2.25, 'GAP + FC\n[B,64]->[B,N_cls]',
            ha='center', va='center', fontsize=10)
    ax.text(10.4, 3.0, '[B,64]', fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # Output
    output_box = FancyBboxPatch((11.6, 1.5), 1.2, 1, boxstyle="round,pad=0.1",
                                facecolor='#98e699', edgecolor='black')
    ax.add_patch(output_box)
    ax.text(12.2, 2.0, 'Output\nLogits\n[B,N_cls]',
            ha='center', va='center', fontsize=11)

    # Forward arrows
    arrows_x = [2.5, 4.2, 6.0, 7.8, 9.6, 11.2]
    arrows_y = [5.5, 5.5, 5.5, 4.5, 3.5, 2.0]
    for i in range(len(arrows_x) - 1):
        ax.annotate('', xy=(arrows_x[i+1], arrows_y[i+1]),
                    xytext=(arrows_x[i], arrows_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))

    # Skip connection rule (annotation)
    ax.text(8.8, 1.0,
            'Skip rule:\n'
            '- identity if channels & stride unchanged\n'
            '- else 1x1 conv with stride s (projection)',
            ha='left', va='bottom', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

    # Legend
    legend_elements = [
        patches.Patch(facecolor='#98e699', edgecolor='black', label='Input / Output'),
        patches.Patch(facecolor='#fff9d6', edgecolor='black', label='Stem Conv'),
        patches.Patch(facecolor='#f4a3a3', edgecolor='black', label='Residual Block'),
        patches.Patch(facecolor='#cfe9f7', edgecolor='black', label='Head'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.03, 0.98))
    plt.tight_layout()

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop_path, "ResNet_Architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Figure 1 saved to: {save_path}")

def create_figure2():
    """Figure 2: Backpropagation through one residual block (layout preserved)"""
    fig, ax = plt.subplots(1, 1, figsize=(14.5, 9.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.4, 'Backward Through One Residual Block: Gradient Split & Merge',
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))

    # Positions (preserved)
    components = {
        'input': (1, 6),
        'conv1': (3, 7),
        'bn1': (3, 6),
        'relu1': (3, 5),
        'conv2': (5, 6),
        'bn2': (5, 5),
        'residual': (5, 4),   # F(x)
        'output': (7, 6),     # x^(l)
        'skip_conv': (3, 3),
        'skip_path': (5, 3)
    }

    # Main path boxes
    def add_box(name, x, y, color, h=0.8):
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black')
        ax.add_patch(box)
        return box

    add_box('input', *components['input'], '#cfe0f7')
    add_box('conv1', *components['conv1'], '#fff9d6')
    add_box('bn1', *components['bn1'], '#c3f0b3')
    add_box('relu1', *components['relu1'], '#ffb347')
    add_box('conv2', *components['conv2'], '#fff9d6')
    add_box('bn2', *components['bn2'], '#c3f0b3')
    add_box('residual', *components['residual'], '#f7b2b2')  # F(x)
    add_box('output', *components['output'], '#cfcfcf')
    add_box('skip_conv', *components['skip_conv'], '#d9d9d9')
    add_box('skip_path', *components['skip_path'], '#d9d9d9')

    # Labels (ASCII only)
    labels = {
        'input': 'X^(l-1)\ninput',
        'conv1': 'Conv1\nW1, b1',
        'bn1': 'BN1\ngamma1, beta1',
        'relu1': 'ReLU',
        'conv2': 'Conv2\nW2, b2',
        'bn2': 'BN2\ngamma2, beta2',
        'residual': 'F(x)',
        'output': 'X^(l)\noutput',
        'skip_conv': 'Skip\n(identity / 1x1, s)',
        'skip_path': 'S(x)'
    }
    for name, (x, y) in components.items():
        ax.text(x, y, labels[name], ha='center', va='center', fontsize=10)

    # Forward arrows (grey)
    fwd = [
        ('input', 'conv1'), ('conv1', 'bn1'), ('bn1', 'relu1'),
        ('relu1', 'conv2'), ('conv2', 'bn2'),
        ('bn2', 'output'),
        ('input', 'skip_conv'), ('skip_conv', 'skip_path'),
        ('skip_path', 'output')
    ]
    for s, e in fwd:
        sx, sy = components[s]
        ex, ey = components[e]
        if s == 'bn2' and e == 'output':
            sy += 0.28
        if s == 'skip_path' and e == 'output':
            sx += 0.28; ex -= 0.22; ey += 0.22
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=1.6, alpha=0.85))

    # Merge point before ReLU
    plus_x, plus_y = 6.2, 6.0
    ax.add_patch(plt.Circle((plus_x, plus_y), 0.14, color='black', fill=False, lw=1.8))
    ax.text(plus_x, plus_y, '+', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(6.5, 6.3, "grads split at '+'", fontsize=10)

    # Backward gradients (red)
    grads = [
        ('output', 'bn2', 'G^(l)'), ('output', 'skip_path', 'G^(l)'),
        ('bn2', 'conv2', 'G_Z2'), ('conv2', 'relu1', 'G_H1'),
        ('relu1', 'bn1', 'G_A1'), ('bn1', 'conv1', 'G_Z1'),
        ('conv1', 'input_main', 'G_X,main'),
        ('skip_path', 'skip_conv', 'G_S^(l)'), ('skip_conv', 'input_skip', 'G_X,skip'),
        (('custom', plus_x, plus_y), 'residual', 'G_F^(l)')
    ]
    for s, e, lab in grads:
        if isinstance(s, tuple) and s[0] == 'custom':
            sx, sy = s[1], s[2]
        else:
            sx, sy = components[s]
        if e == 'input_main':
            ex, ey = 1.5, 6.2
        elif e == 'input_skip':
            ex, ey = 1.5, 5.8
        else:
            ex, ey = components[e]
        if e == 'residual':  # from '+' down to F(x)
            ey = ey + 0.50
        ax.annotate('', xytext=(sx, sy), xy=(ex, ey),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.0, alpha=0.9))
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my, lab, fontsize=9, color='red',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.85))

    # Gradient sum at block input
    sum_circle = plt.Circle((1.5, 6), 0.15, fill=True, color='red', alpha=0.75)
    ax.add_patch(sum_circle)
    ax.text(1.5, 6, '+', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    ax.text(1.85, 6, 'dL/dX^(l-1) = G_X,main + G_X,skip',
            ha='left', va='center', fontsize=11, color='red', fontweight='bold')

    # Legend and key points
    legend_elements = [
        patches.Patch(facecolor='#fff9d6', edgecolor='black', label='Conv'),
        patches.Patch(facecolor='#c3f0b3', edgecolor='black', label='BatchNorm'),
        patches.Patch(facecolor='#ffb347', edgecolor='black', label='Activation'),
        patches.Patch(facecolor='#f7b2b2', edgecolor='black', label='Residual F(x)'),
        patches.Patch(facecolor='#d9d9d9', edgecolor='black', label='Skip'),
        plt.Line2D([0], [0], color='grey', lw=2, label='Forward'),
        plt.Line2D([0], [0], color='red', lw=2, label='Backward grads')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.97, 0.98))

    key_text = (
        "Key points:\n"
        "• Gradient splits at '+': G_F(l) = G_res(l), G_S(l) = G_res(l)\n"
        "• Two paths backprop independently\n"
        "• At block input, gradients add: dL/dX^(l-1) = G_X,main + G_X,skip\n"
        "• Identity path offers a direct route that helps against vanishing gradients"
    )
    ax.text(8.2, 7.0, key_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
            va='top', ha='left')

    plt.tight_layout()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop_path, "ResNet_Backward_Pass.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Figure 2 saved to: {save_path}")

if __name__ == "__main__":
    print("Generating Figure 1...")
    create_figure1()
    print("\nGenerating Figure 2...")
    create_figure2()
    print("\nAll figures have been generated and saved on your Desktop.")
