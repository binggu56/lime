# Use - to go back to previous directory
alias -- -='cd -'

# Taken from the tmux plugin
alias ta="tmux attach -t"
alias ts="tmux new-session -s"
alias tl="tmux list-sessions"

# Keybindings

autoload -U up-line-or-beginning-search
autoload -U down-line-or-beginning-search

# [Space] - do history expansion
bindkey ' ' magic-space

# start typing + [Up-Arrow] - fuzzy find history forward
bindkey "${terminfo[kcuu1]}" up-line-or-beginning-search

# start typing + [Down-Arrow] - fuzzy find history backward
bindkey "${terminfo[kcud1]}" down-line-or-beginning-search
