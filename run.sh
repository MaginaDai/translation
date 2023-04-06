# python main.py -name 'attention' -lr 0.005 -e 30 &
# python main.py -name 'attention' -lr 0.005 -e 50 &
# python main.py -name 'attention' -lr 0.005 -e 75 &
# python main.py -name 'attention' -lr 0.005 -e 100 &

# python main.py -name 'multiplicative_attention' -lr 0.01 -e 30
python main.py -name 'multiplicative_attention' -lr 0.001 -e 20 -load 'runs/multiplicative_attention_attention_e30_lr0.001/model.pt'
