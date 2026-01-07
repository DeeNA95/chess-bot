var board = null
var game = new Chess()
var $status = $('#game-status')
var $fen = $('#fen')
var $pgn = $('#pgn')
var $eval = $('#eval')

function onDragStart (source, piece, position, orientation) {
  // do not pick up pieces if the game is over
  if (game.game_over()) return false

  // only pick up pieces for the side to move
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false
  }
}

function onDrop (source, target) {
  // see if the move is legal
  var move = game.move({
    from: source,
    to: target,
    promotion: 'q' // NOTE: always promote to a queen for example simplicity
  })

  // illegal move
  if (move === null) return 'snapback'

  updateStatus()

  // Make bot move
  if (!game.game_over()) {
      makeBotMove()
  }
}

function makeBotMove() {
    $status.html("Bot thinking...")
    $.ajax({
        url: 'http://localhost:8000/move',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ fen: game.fen() }),
        success: function(response) {
            game.move(response.uci, {
                sloppy: true
            })
            board.position(game.fen())
            $eval.text(response.evaluation.toFixed(2))
            updateStatus()
        },
        error: function(error) {
            console.error(error)
            $status.html("Error: " + error.responseText)
        }
    })
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd () {
  board.position(game.fen())
}

function updateStatus () {
  var status = ''

  var moveColor = 'White'
  if (game.turn() === 'b') {
    moveColor = 'Black'
  }

  // checkmate?
  if (game.in_checkmate()) {
    status = 'Game over, ' + moveColor + ' is in checkmate.'
  }

  // draw?
  else if (game.in_draw()) {
    status = 'Game over, drawn position'
  }

  // game still on
  else {
    status = moveColor + ' to move'

    // check?
    if (game.in_check()) {
      status += ', ' + moveColor + ' is in check'
    }
  }

  $status.html(status)
  $pgn.html(game.pgn())
}

var config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
}
board = Chessboard('myBoard', config)

updateStatus()

$('#resetBtn').on('click', function() {
    game.reset()
    board.start()
    updateStatus()
    $eval.text("0.0")
})

$('#flipBtn').on('click', board.flip)
