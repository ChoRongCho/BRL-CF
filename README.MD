# POMCP Algorithm

Required packages:

```latex
\normalem  % remove auto underline
\usepackage{graphicx}
\usepackage{algpseudocode}
\usepackage[linesnumbered,ruled,vlined]{algorithm2e}
```

Algorithm:

```latex
\begin{algorithm}[t]
\caption{POMCP Search with Applicable Actions}\label{alg:pomcp}
\textbf{Input}: belief $b$ \\
\textbf{Output}: a selected action $a \in \mathcal{A}$
\BlankLine

\SetKwProg{Proc}{procedure}{}{}
\SetKwFunction{Search}{SEARCH}
\SetKwFunction{SampleParticle}{SAMPLE-PARTICLE}
\SetKwFunction{Simulate}{SIMULATE}
\SetKwFunction{SearchBest}{SEARCH-BEST}
\SetKwFunction{UCB}{UCB}
\SetKwFunction{Rollout}{ROLLOUT}

\Proc{\Search{$b$}}{
    $h \gets$ current root history\;
    $k \gets \kappa(b)$\;
    \For{$i \gets 1$ \KwTo $n_{\mathrm{sim}}$}{
        $s \gets$ \SampleParticle{$h, k$}\;
        \Simulate{$s, h, 0$}\;
    }
    \Return \SearchBest{$h, k, \textnormal{false}$}\;
}
\BlankLine

\Proc{\SampleParticle{$h, k$}}{
    \eIf{$\mathcal{B}(h) \neq \emptyset$}{
        sample $s \sim \mathcal{B}(h)$\;
        \Return $s$\;
    }{
        \Return $k$\;
    }
}
\BlankLine

\Proc{\Simulate{$s, h, d$}}{
    \If{$(\gamma^d < \epsilon \lor \gamma = 0) \land d > 0$}{
        \Return $0$\;
    }
    \If{$\mathcal{A}(s) = \emptyset$}{
        $N(h) \gets N(h) + 1$\;
        \Return $0$\;
    }
    \If{$h$ is unexpanded in $\mathcal{T}$}{
        \ForEach{$a \in \mathcal{A}(s)$}{
            $\mathcal{T} \gets \mathcal{T} \cup \{ha\}$\;
            $N(ha) \gets 0$\;
            $Q(ha) \gets 0$\;
        }
        $G \gets$ \Rollout{$s, d$}\;
        $N(h) \gets N(h) + 1$\;
        \Return $G$\;
    }
    $a \gets$ \SearchBest{$h, s, \textnormal{true}$}\;
    $s' \sim P(\cdot \mid s, a)$\;
    $o \sim O(\cdot \mid s', a)$\;
    $r \gets R(s, a, s')$\;
    $G \gets r + \gamma \cdot$ \Simulate{$s', hao, d+1$}\;
    $\mathcal{B}(h) \gets \mathcal{B}(h) \cup \{s\}$\;
    $N(h) \gets N(h) + 1$\;
    $N(ha) \gets N(ha) + 1$\;
    $Q(ha) \gets Q(ha) + \frac{G - Q(ha)}{N(ha)}$\;
    \Return $G$\;
}
\BlankLine

\Proc{\SearchBest{$h, s, use\_ucb$}}{
    $\mathcal{C} \gets \{a \in \mathcal{A}(s) \mid ha \in \mathcal{T}\}$\;
    \If{$\mathcal{C} = \emptyset$}{
        \Return $\emptyset$\;
    }
    \eIf{$use\_ucb$}{
        \Return $\arg\max_{a \in \mathcal{C}}$ \UCB{$h, a$}\;
    }{
        \Return $\arg\max_{a \in \mathcal{C}} Q(ha)$\;
    }
}
\BlankLine

\Proc{\UCB{$h, a$}}{
    \If{$N(ha) = 0$}{
        \Return $\infty$\;
    }
    \Return $Q(ha) + c \cdot \frac{\sqrt{\max(1, N(h))}}{1 + N(ha)}$\;
}
\BlankLine

\Proc{\Rollout{$s, d$}}{
    \If{$(\gamma^d < \epsilon \lor \gamma = 0) \land d > 0$}{
        \Return $0$\;
    }
    \If{$\mathcal{A}(s) = \emptyset$}{
        \Return $0$\;
    }
    $a \sim \mathrm{Uniform}(\mathcal{A}(s))$\;
    $s' \sim P(\cdot \mid s, a)$\;
    $r \gets R(s, a, s')$\;
    \Return $r + \gamma \cdot$ \Rollout{$s', d+1$}\;
}
\end{algorithm}
```

Notation:

- \(b\): current belief.
- \(\kappa(b)\): symbolic knowledge state summarized from \(b\).
- \(h\): action-observation history.
- \(ha\): history extended by action \(a\).
- \(hao\): history extended by action \(a\) and observation \(o\).
- \(\mathcal{B}(h)\): particle belief associated with history \(h\).
- \(\mathcal{A}(s)\): applicable action set in state \(s\).
- \(\mathcal{T}\): search tree.
- \(N(h)\): visit count of history \(h\).
- \(Q(ha)\): estimated action value after taking \(a\) at history \(h\).
- \(P(s' \mid s,a)\): transition model.
- \(O(o \mid s',a)\): observation model.
- \(R(s,a,s')\): reward model.

## RP Description for System Architecture

```latex
The \textit{Reactive Planner} (RP) is responsible for online action selection
and execution under partial observability. In our implementation, the RP
corresponds to the execution pipeline in \texttt{main.py}. At each execution
step, the RP receives the current belief state, invokes the POMCP-based planner
to select an action, executes the selected action in the environment, and then
updates the belief using the resulting observation. After the belief update, the
RP prunes the search tree according to the executed action and the received
observation, so that subsequent planning is rooted at the latest
action-observation history.

This online loop enables the RP to reason over uncertain local information
maintained in the KB while continuously incorporating new observations from the
SP module. Instead of committing to a fixed offline plan, the RP repeatedly
replans from the updated belief, allowing the robot to adapt its behavior when
the perceived state differs from the previously expected state. When the
confidence of task-relevant local information is insufficient, the RP can query
the FM to verify the corresponding symbolic fact before continuing execution.
```
