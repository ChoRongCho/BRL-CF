% Domain: Blocksworld
% Meta data
% description: Generate all possible worlds consistent with partial observations in Blocksworld
% semantics: ASP-style world generation under partial observability
% purpose: belief state initialization / possible world enumeration
% assumptions: ['Every block is on exactly one support', 'A support is either table or another block', 'A block cannot support itself', 'Two different blocks cannot be on the same block simultaneously', 'Observed relations are hard evidence', 'Unobserved blocks may be assigned anywhere consistent with constraints']

% Runtime facts
block(b1).
block(b2).
block(b3).
block(b4).
block(b5).
block(b6).
block(b7).
visible(b1).
visible(b2).
visible(b3).
observed_on(b1,table).
observed_on(b2,b1).
observed_on(b3,table).

% Domain facts / declarations
block(B).
support(S).
hidden(B).
visible(B).
observed_on(B, S).
on(B, S).
clear(X).
above(B1, B2).
possible_support(B, S).
block_count(N).
visible_count(N).
hidden_count(N).

% Derived rules
support(table).
support(B) :- block(B).
visible(B) :- observed_on(B, S).
hidden(B) :- block(B), not visible(B).
on(B, S) :- observed_on(B, S).
possible_support(B, table) :- hidden(B).
possible_support(B, S) :- hidden(B), block(S), B != S.
above(B1, B2) :- on(B1, B2).
above(B1, B3) :- on(B1, B2), above(B2, B3).
clear(S) :- support(S), not occupied(S).
occupied(S) :- on(B, S).
hidden_count(N) :- N = #count { B : hidden(B) }.
visible_count(N) :- N = #count { B : visible(B) }.

% Choice rules
1 { on(B, S) : possible_support(B, S) } 1 :- hidden(B).

% Transition / generic rules
world_consistent :- not inconsistent.
inconsistent :- on(B, B).
inconsistent :- above(B, B).
inconsistent :- observed_on(B, S1), on(B, S2), S1 != S2.

% Constraints
:- on(B, B).
:- on(B1, S), on(B2, S), block(S), B1 != B2.
:- above(B, B).
:- observed_on(B, S1), on(B, S2), S1 != S2.
:- visible(B), not 1 { on(B, S) : support(S) } 1.
:- hidden(B), not 1 { on(B, S) : support(S) } 1.
:- on(B1, B2), visible(B2), clear(B2), observed_on(B2, S).

% Show directives
#show on/2.
#show hidden/1.
#show visible/1.
#show clear/1.
