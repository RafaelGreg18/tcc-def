from pushbullet import Pushbullet

# Substitua pelo seu token
pb = Pushbullet("o.QAfxjrtfUV6tyWkVpae2oEeXaSPT0aGP")

# Função para enviar notificações
def enviar_notificacao(titulo, mensagem):
    push = pb.push_note(titulo, mensagem)
    print(f"Notificação enviada: {titulo} - {mensagem}")

# Exemplo de uso
enviar_notificacao("Modelo 1 Seed 1", "Simulação concluída!")
