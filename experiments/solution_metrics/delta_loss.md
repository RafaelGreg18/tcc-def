# Experimento 1

## Pergunta

A **queda média (suavizada)** da perda **no dataset de treino** por rodada é um bom **sinal para aumentar ou diminuir** o número de clientes selecionados (Cₜ) no FL?

## Hipótese

H₁: Quando a queda média local **fica pequena**, **aumentar Cₜ** na próxima rodada melhora o desempenho final; quando a queda **fica grande**, **reduzir Cₜ** não piora (ou melhora por reduzir ruído/heterogeneidade).
H₀: Adaptar Cₜ com base nessa queda **não** supera esquemas com C fixo.

## Definições do sinal (no cliente i, rodada t)

* Antes de treinar com o modelo global $w_t$, o cliente computa a perda média de **treino**: $L^{\text{pre}}_{i,t}$.
* Após E épocas locais, computa $L^{\text{post}}_{i,t}$.
* **Queda absoluta:** $r_{i,t}=L^{\text{pre}}_{i,t}-L^{\text{post}}_{i,t}$.
* **Queda relativa (recomendada):** $\rho_{i,t}=\frac{r_{i,t}}{\max(L^{\text{pre}}_{i,t},\varepsilon)}$.
* **Agregação robusta no servidor:**
  $A_t=\text{média aparada 10\%}$ das $\rho_{i,t}$, **ponderada por** $n_i$ (tamanho do dataset local).
* **Suavização temporal:** $S_t=\text{EMA}_\beta(A_t)$ com $\beta=0{,}6$.

## Métricas de avaliação

* **Primária (só no fim):** $L_{\text{test}}$ após T rodadas.
* **Secundárias:** melhor $L_{\text{test}}$ ao longo do treino; rodadas até “platô” (definido por proxy de treino, ver abaixo); área sob a curva do **proxy de perda global de treino**; variabilidade entre seeds.

## Tratamentos

1. **Adaptive-S (proposto):**

   * **Warm-up**: 10 rodadas com $C=C_{\text{mid}}$. Colete $S_t$.
   * Defina **limiares**: $\tau_{\text{up}} = P75(S_{1:10})$ e $\tau_{\text{down}} = P25(S_{1:10})$.
   * Para $t\ge 11$:

     * Se $S_t \le \tau_{\text{down}}$ ⇒ **aumenta**: $C_{t+1}\leftarrow \min(C_t+s_{\uparrow},C_{\max})$.
     * Se $S_t \ge \tau_{\text{up}}$ ⇒ **reduz**: $C_{t+1}\leftarrow \max(C_t-s_{\downarrow},C_{\min})$.
     * Caso contrário: mantém $C_{t+1}=C_t$.
   * Sugestões: $C_{\min}=5,\ C_{\text{mid}}=15,\ C_{\max}=25;\ s_{\uparrow}=s_{\downarrow}=5$.
2. **Baselines fixos:** $C\in\{C_{\min},C_{\text{mid}},C_{\max}\}$.
3. **Baseline aleatório (mesma trajetória de C):** mesmo número de aumentos/reduções do Adaptive-S, mas em rodadas sorteadas.

## Procedimento

* **Dados:** K=100 clientes; particionamento não-IID (ex.: Dirichlet α=0.3). **Sem validação central**; **teste central** só ao final.
* **Treino:** FedAvg; E=5; T=100; LR fixo; mesma inicialização para todos os tratamentos.
* **Seeds:** 5 execuções por condição.
* **Coleta por rodada:** $\{L^{\text{pre}}_{i,t},L^{\text{post}}_{i,t},n_i\}$, $A_t$, $S_t$, $C_t$.

### Proxy de progresso (sem teste)

Para checar previsibilidade do sinal sem tocar no teste:

* **Perda global de treino estimada:** em cada rodada, um **subconjunto hold-out de m clientes** (não necessariamente participantes) reporta a perda média de treino do **modelo global $w_t$** em seus dados.
* Agregue ponderado por $n_i$ ⇒ $\tilde{L}^{\text{train}}_t$.
* **Ganho futuro (proxy):** $\Delta^+_t=\tilde{L}^{\text{train}}_{t}-\tilde{L}^{\text{train}}_{t+1}$.

## Análise

1. **Efeito no desfecho:** compare $L_{\text{test}}$ final (Adaptive-S vs baselines) com **teste t pareado** (ou Wilcoxon) entre seeds; reporte média±dp e **Cohen’s d**.
2. **Validade preditiva do sinal:**

   * **Spearman** entre $S_t$ e $\Delta^+_t$ (esperado: correlação **positiva** — S alto ⇒ mais ganho no próximo passo — ou negativa se você definir S como “queda” negativa; mantenha o sinal consistente).
   * **Regressão logística**: prever “**vale a pena aumentar C?**” a partir de $S_t$. Rotule como 1 se, nas próximas k rodadas, $\sum_{j=1}^k \Delta^+_{t+j} > \eta$. Reporte **AUC ROC**.
3. **Critérios de sucesso:** rejeitar H₀ se Adaptive-S tiver $L_{\text{test}}$ final **significativamente menor** que todos os C fixos **e** AUC > 0.6.

## Ameaças à validade (mitigação)

* **Overfitting local**: use **queda relativa $\rho$** e agregação **robusta** (média aparada/mediana ponderada).
* **Ruído por rodada**: suavize com **EMA** e use limiares por **percentis** do warm-up.
* **Não-IID forte**: repita com outro α ou outro dataset (fase 2).
* **Custo de medir $L^{\text{pre}}$/$L^{\text{post}}$**: se caro, use um **subset fixo** por cliente (mesma amostra a cada rodada) para estimar as perdas.

## Relato mínimo

* Curvas de $S_t$ e $C_t$ por rodada; distribuição de $\rho_{i,t}$ (boxplots); $\tilde{L}^{\text{train}}_t$ vs tempo; tabela com $L_{\text{test}}$ final (média±dp, p-values, d).