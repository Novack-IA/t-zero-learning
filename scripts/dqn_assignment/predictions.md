Previsões registradas antes de rodar as varreduras (protocolo da Parte 3).

Q1 (target_network_frequency). Com freq=1 o alvo deixa de ser um alvo fixo e passa a
se mover junto com a rede online a cada passo; espero realimentação positiva sobre o
próprio erro, q_values subindo de forma descontrolada e retorno instável ou em colapso.
Com freq muito grande (5000) o alvo fica quase congelado: o aprendizado deve ficar
lento, com q_values em degraus e td_loss com picos a cada sincronização. O valor
intermediário (500) deve ser o melhor.

Q2 (buffer_size). Com buffer minúsculo (500) o minibatch de 128 vem de uma janela de
poucos episódios quase consecutivos, logo as amostras são fortemente correlacionadas e
a rede sofre esquecimento catastrófico das situações antigas. Espero retorno baixo e
oscilante, e q_values seguindo a política recente em vez de convergir. Buffer grande
(200000) deve ser igual ou levemente melhor que o baseline no CartPole.

Q3 (exploration_fraction). Com fração 0,02 o epsilon cai para 0,05 quase imediatamente,
antes de o buffer conter dados variados; espero política gulosa cedo demais e retorno
preso num valor baixo. Com 0,9 o agente passa quase todo o treino agindo aleatoriamente
em 30 a 50 por cento dos passos, então o retorno durante o treino fica limitado pelo
ruído da própria exploração, ainda que a rede aprenda.
